"""Audit the code-only public release surface declared by its inventory.

Workspace mode reports files that still need to be added to Git. Release mode
turns that report into an error. Neither mode stages, commits, or pushes files.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY: Final[Path] = Path(__file__).with_name("public_release_inventory.json")
STDLIB: Final[frozenset[str]] = frozenset(sys.stdlib_module_names)


class ReleaseAuditError(RuntimeError):
    """Raised for a malformed inventory or an unsafe release surface."""


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ReleaseAuditError(f"JSON root must be an object: {path}")
    return value


def _normalise_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ReleaseAuditError(f"inventory path must stay under the repository: {value!r}")
    return path.as_posix()


def inventory_files(inventory: Mapping[str, Any]) -> list[str]:
    categories = inventory.get("categories")
    if not isinstance(categories, Mapping):
        raise ReleaseAuditError("inventory.categories must be an object")
    files: list[str] = []
    for category, values in categories.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ReleaseAuditError(f"inventory category {category!r} must be an array")
        files.extend(_normalise_relative(str(value)) for value in values)
    duplicates = sorted({path for path in files if files.count(path) > 1})
    if duplicates:
        raise ReleaseAuditError(f"duplicate inventory paths: {duplicates}")
    return files


def forbidden_release_paths(
    paths: Iterable[str],
    *,
    prefixes: Sequence[str],
    suffixes: Sequence[str],
    patterns: Sequence[str] = (),
) -> list[str]:
    """Return release paths that look like private data or generated artifacts."""

    normalised_prefixes = tuple(_normalise_relative(str(value)).rstrip("/") + "/" for value in prefixes)
    normalised_suffixes = tuple(str(value).casefold() for value in suffixes)
    compiled_patterns = [re.compile(str(value), flags=re.I) for value in patterns]
    violations: list[str] = []
    for raw in paths:
        path = _normalise_relative(str(raw))
        folded = path.casefold()
        if (
            path.startswith(normalised_prefixes)
            or folded.endswith(normalised_suffixes)
            or any(pattern.search(path) for pattern in compiled_patterns)
        ):
            violations.append(path)
    return sorted(set(violations))


def category_suffix_violations(inventory: Mapping[str, Any]) -> list[str]:
    """Enforce code-only extensions for security-sensitive release categories."""

    categories = inventory.get("categories")
    policies = inventory.get("category_allowed_suffixes", {})
    if not isinstance(categories, Mapping):
        raise ReleaseAuditError("inventory.categories must be an object")
    if not isinstance(policies, Mapping):
        raise ReleaseAuditError("inventory.category_allowed_suffixes must be an object")
    violations: list[str] = []
    for category, raw_suffixes in policies.items():
        if category not in categories:
            raise ReleaseAuditError(
                f"category suffix policy refers to unknown category: {category!r}"
            )
        if not isinstance(raw_suffixes, Sequence) or isinstance(raw_suffixes, (str, bytes)):
            raise ReleaseAuditError(
                f"allowed suffixes for {category!r} must be an array"
            )
        suffixes = tuple(str(value).casefold() for value in raw_suffixes)
        if not suffixes or any(not value.startswith(".") for value in suffixes):
            raise ReleaseAuditError(
                f"allowed suffixes for {category!r} must be non-empty extensions"
            )
        for raw_path in categories[category]:
            path = _normalise_relative(str(raw_path))
            if not path.casefold().endswith(suffixes):
                violations.append(f"{category}: {path}")
    return sorted(violations)


def _module_path(root: Path, module: str, search_roots: Sequence[Path] = ()) -> Path | None:
    parts = module.split(".")
    for base in (root, *search_roots):
        module_file = base.joinpath(*parts).with_suffix(".py")
        if module_file.is_file():
            return module_file
        package_file = base.joinpath(*parts, "__init__.py")
        if package_file.is_file():
            return package_file
    return None


def _imports(path: Path) -> list[str]:
    # ``utf-8-sig`` accepts ordinary UTF-8 and the BOM carried by a few legacy
    # production modules in this repository.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module)
            if node.module in {"experiments", "models", "services", "test", "utils"}:
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def dependency_closure(
    root: Path,
    entrypoints: Iterable[str],
    local_roots: set[str],
    local_search_roots: Sequence[str] = (),
) -> tuple[set[str], set[str], list[str]]:
    search_roots = tuple(root / _normalise_relative(value) for value in local_search_roots)
    queue: deque[Path] = deque(root / _normalise_relative(value) for value in entrypoints)
    visited: set[Path] = set()
    external: set[str] = set()
    unresolved: list[str] = []
    while queue:
        path = queue.popleft().resolve()
        if path in visited:
            continue
        visited.add(path)
        if not path.is_file():
            unresolved.append(path.relative_to(root).as_posix())
            continue
        if path.suffix != ".py":
            continue
        for module in _imports(path):
            top = module.split(".", 1)[0]
            local = _module_path(root, module, search_roots)
            if local is not None:
                queue.append(local)
            elif top in local_roots:
                # A package-level symbol is not necessarily a submodule. Only
                # report the full import when its top-level package also fails.
                if _module_path(root, top, search_roots) is None and not (root / top).is_dir():
                    unresolved.append(f"{path.relative_to(root).as_posix()}: {module}")
            elif top not in STDLIB and top != "__future__":
                external.add(top)
    relative = {path.relative_to(root).as_posix() for path in visited if path.is_file()}
    return relative, external, sorted(set(unresolved))


def _locked_distributions(path: Path) -> set[str]:
    values: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)==", line)
        if match:
            values.add(match.group(1).lower().replace("_", "-"))
    return values


def _git_lines(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _readme_links(root: Path) -> list[str]:
    text = (root / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    missing: list[str] = []
    for target in links:
        clean = target.split("#", 1)[0].strip().strip("<>")
        if not clean or re.match(r"^[a-z]+://", clean, flags=re.I):
            continue
        if not (root / clean).exists():
            missing.append(clean)
    return sorted(set(missing))


def run_audit(root: Path, inventory_path: Path, *, mode: str) -> Mapping[str, Any]:
    inventory = _load_json(inventory_path)
    if inventory.get("schema_version") != 1:
        raise ReleaseAuditError("unsupported public release inventory schema")
    declared = inventory_files(inventory)
    missing = sorted(path for path in declared if not (root / path).is_file())

    entrypoints = inventory.get("python_entrypoints", [])
    local_roots = {str(value) for value in inventory.get("local_module_roots", [])}
    closure, external, unresolved = dependency_closure(
        root,
        entrypoints,
        local_roots,
        [str(value) for value in inventory.get("local_import_search_roots", [])],
    )
    undeclared_dependencies = sorted(closure.difference(declared))

    module_map = inventory.get("external_modules", {})
    optional_external = {str(value) for value in inventory.get("optional_external_modules", [])}
    unknown_external = sorted(
        external.difference(str(key) for key in module_map).difference(optional_external)
    )
    lock_path = root / "experiments/requirements-training.lock.txt"
    locked = _locked_distributions(lock_path)
    missing_distributions = sorted(
        {
            str(module_map[module]).lower().replace("_", "-")
            for module in external.intersection(module_map)
            if str(module_map[module]).lower().replace("_", "-") not in locked
        }
    )

    tracked = set(_git_lines(root, "ls-files"))
    untracked_required = sorted(path for path in declared if path not in tracked)
    tracked_paths = sorted(tracked)
    forbidden_prefixes = [
        str(value) for value in inventory.get("forbidden_tracked_prefixes", [])
    ]
    forbidden_suffixes = [
        str(value).casefold()
        for value in inventory.get("forbidden_tracked_suffixes", [])
    ]
    forbidden_path_patterns = [
        str(value)
        for value in inventory.get("forbidden_declared_path_patterns", [])
    ]
    forbidden_declared = forbidden_release_paths(
        declared,
        prefixes=forbidden_prefixes,
        suffixes=forbidden_suffixes,
        patterns=forbidden_path_patterns,
    )
    forbidden_tracked = forbidden_release_paths(
        tracked_paths,
        prefixes=forbidden_prefixes,
        suffixes=forbidden_suffixes,
        patterns=forbidden_path_patterns,
    )
    invalid_category_suffixes = category_suffix_violations(inventory)

    ignore_text = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    ignore_rules = {line.strip() for line in ignore_text if line.strip() and not line.lstrip().startswith("#")}
    missing_ignore_rules = sorted(
        str(rule) for rule in inventory.get("required_gitignore_patterns", []) if str(rule) not in ignore_rules
    )

    # The inventory defines the new research surface, but an existing tracked
    # service file would still ship in the same public repository. Scan both.
    content_files = sorted({*declared, *closure, *tracked})
    forbidden_content: list[str] = []
    patterns = [re.compile(str(value)) for value in inventory.get("forbidden_content_patterns", [])]
    for relative in content_files:
        path = root / relative
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            if pattern.search(text):
                forbidden_content.append(f"{relative}: {pattern.pattern}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    missing_readme_paths = sorted(
        str(path) for path in inventory.get("readme_required_paths", []) if str(path) not in readme
    )
    missing_readme_links = _readme_links(root)

    errors = {
        "missing_files": missing,
        "unresolved_local_imports": unresolved,
        "undeclared_local_dependencies": undeclared_dependencies,
        "unknown_external_modules": unknown_external,
        "missing_locked_distributions": missing_distributions,
        "forbidden_declared_files": forbidden_declared,
        "forbidden_tracked_files": forbidden_tracked,
        "invalid_category_suffixes": invalid_category_suffixes,
        "missing_gitignore_rules": missing_ignore_rules,
        "forbidden_content": forbidden_content,
        "missing_readme_paths": missing_readme_paths,
        "missing_readme_links": missing_readme_links,
    }
    if mode == "release":
        errors["untracked_required_files"] = untracked_required
    failed = {name: values for name, values in errors.items() if values}
    return {
        "protocol": "pointer_meter_public_code_release_audit_v1",
        "status": "pass" if not failed else "fail",
        "mode": mode,
        "declared_files": len(declared),
        "dependency_closure_files": len(closure),
        "external_modules": sorted(external),
        "untracked_required_files": untracked_required,
        "warnings": (
            {"untracked_required_files": untracked_required}
            if mode == "workspace" and untracked_required
            else {}
        ),
        "errors": failed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--mode", choices=("workspace", "release"), default="workspace")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    report = run_audit(ROOT, args.inventory.resolve(), mode=args.mode)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
