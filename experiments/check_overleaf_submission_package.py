"""Inventory and fail-closed readiness checks for the local Overleaf source tree.

This checker does not compile LaTeX, create an archive, or alter the manuscript.
It reports every blocking issue as JSON and exits with status 2 when the source
tree is not ready for a final package build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_ROOT = PROJECT_ROOT / "paper/submission_mdpi/official"
DEFAULT_PAPER_RESULTS_ROOT = Path(r"C:\pointer_read\paper_final_results_v2")
GENERATED_PROTOCOL = "paper_result_latex_tables_v1"
GENERATED_FILES = (
    "strict-main-table.tex",
    "paired-bootstrap-table.tex",
    "component-oof-table.tex",
    "sensitivity-table.tex",
)


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_object,
        )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tex_argument_values(command: str, text: str) -> list[str]:
    return re.findall(r"\\" + re.escape(command) + r"(?:\[[^]]*\])?\{([^}]+)\}", text)


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", text))


def _citation_keys(text: str) -> set[str]:
    result: set[str] = set()
    for payload in re.findall(r"\\cite\w*(?:\[[^]]*\])?\{([^}]+)\}", text):
        result.update(item.strip() for item in payload.split(",") if item.strip())
    return result


def _validate_generated(
    root: Path,
    issues: list[str],
    expected_paper_results_root: Path | None = None,
) -> dict[str, Any]:
    generated = root / "generated_tables"
    required = [generated / name for name in (*GENERATED_FILES, "manifest.json", "seal.json")]
    missing = [str(path.relative_to(root)).replace("\\", "/") for path in required if not path.is_file()]
    if missing:
        issues.append("missing generated table artifacts: " + ", ".join(missing))
        return {"status": "missing", "missing": missing}
    try:
        manifest = strict_json(generated / "manifest.json")
        seal = strict_json(generated / "seal.json")
        if manifest.get("protocol") != GENERATED_PROTOCOL or manifest.get("status") != "complete":
            issues.append("generated table manifest protocol/status drift")
        if expected_paper_results_root is not None:
            expected_source = expected_paper_results_root.resolve()
            source = manifest.get("source")
            if not isinstance(source, dict):
                issues.append("generated table manifest has no result-source binding")
            else:
                try:
                    observed_source = Path(str(source.get("path") or "")).resolve()
                except (OSError, ValueError):
                    observed_source = Path()
                if observed_source != expected_source:
                    issues.append("generated tables are not bound to the expected v2 result root")
                for name in ("summary", "seal"):
                    source_path = expected_source / f"{name}.json"
                    digest_key = f"{name}_sha256"
                    if not source_path.is_file():
                        issues.append(f"expected v2 paper-result artifact is absent: {name}.json")
                    elif source.get(digest_key) != sha256_file(source_path):
                        issues.append(f"generated table {name} source hash drift")
        if seal.get("protocol") != GENERATED_PROTOCOL or seal.get("status") != "sealed":
            issues.append("generated table seal protocol/status drift")
        artifacts = seal.get("artifacts")
        if not isinstance(artifacts, dict):
            issues.append("generated table seal has no artifact map")
            artifacts = {}
        expected_names = {
            "main_table",
            "paired_group_bootstrap",
            "component_table",
            "sensitivity_table",
            "manifest",
        }
        if set(artifacts) != expected_names:
            issues.append("generated table seal artifact roster drift")
        for name, binding in artifacts.items():
            if not isinstance(binding, dict):
                issues.append(f"generated table seal binding malformed: {name}")
                continue
            relative = binding.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                issues.append(f"unsafe generated table path: {name}")
                continue
            path = generated / relative
            if not path.is_file():
                issues.append(f"generated table artifact absent: {name}")
                continue
            if binding.get("sha256") != sha256_file(path):
                issues.append(f"generated table hash drift: {name}")
            if int(binding.get("bytes", -1)) != path.stat().st_size:
                issues.append(f"generated table size drift: {name}")
        if artifacts and seal.get("bundle_sha256") != canonical_sha256(artifacts):
            issues.append("generated table bundle hash drift")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        issues.append(f"cannot authenticate generated tables: {exc}")
    return {"status": "present", "files": [path.name for path in required]}


def inspect_package(
    root: Path,
    expected_paper_results_root: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    issues: list[str] = []
    required = [
        root / "manuscript.tex",
        root / "references.bib",
        root / "Definitions/mdpi.cls",
        root / "Definitions/journalnames.tex",
    ]
    missing_required = [
        str(path.relative_to(root)).replace("\\", "/") for path in required if not path.is_file()
    ]
    if missing_required:
        issues.append("missing required package files: " + ", ".join(missing_required))

    generated = _validate_generated(root, issues, expected_paper_results_root)
    manuscript_path = root / "manuscript.tex"
    bib_path = root / "references.bib"
    manuscript = manuscript_path.read_text(encoding="utf-8") if manuscript_path.is_file() else ""
    bibliography = bib_path.read_text(encoding="utf-8") if bib_path.is_file() else ""

    expected_class = r"\documentclass[electronics,article,submit,moreauthors]{Definitions/mdpi}"
    class_ok = expected_class in manuscript
    if manuscript and not class_ok:
        issues.append("Electronics MDPI documentclass is absent or changed")

    todo_invocations = len(re.findall(r"\\todo\{", manuscript))
    if r"\newcommand{\todo}" in manuscript and todo_invocations:
        todo_invocations -= 1
    todo_lines = [
        index
        for index, line in enumerate(manuscript.splitlines(), 1)
        if "TODO" in line.upper()
        and not line.lstrip().startswith("%")
        and r"\newcommand{\todo}" not in line
    ]
    if todo_invocations or todo_lines:
        issues.append(
            f"unresolved TODO markers: {todo_invocations} macro invocation(s), lines {todo_lines}"
        )

    citations = _citation_keys(manuscript)
    bib_keys = _bib_keys(bibliography)
    missing_bib = sorted(citations - bib_keys)
    if missing_bib:
        issues.append("missing BibTeX entries: " + ", ".join(missing_bib))

    labels = re.findall(r"\\label\{([^}]+)\}", manuscript)
    references = set(re.findall(r"\\(?:ref|pageref|autoref)\{([^}]+)\}", manuscript))
    undefined_references = sorted(references - set(labels))
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if undefined_references:
        issues.append("undefined LaTeX references: " + ", ".join(undefined_references))
    if duplicate_labels:
        issues.append("duplicate LaTeX labels: " + ", ".join(duplicate_labels))

    graphics = _tex_argument_values("includegraphics", manuscript)
    missing_figures = [name for name in graphics if not (root / name).is_file()]
    if not graphics:
        issues.append("manuscript contains no figure assets")
    if missing_figures:
        issues.append("missing referenced figures: " + ", ".join(missing_figures))

    inputs = set(_tex_argument_values("input", manuscript))
    expected_inputs = {f"generated_tables/{Path(name).stem}" for name in GENERATED_FILES}
    normalized_inputs = {value[:-4] if value.endswith(".tex") else value for value in inputs}
    missing_inputs = sorted(expected_inputs - normalized_inputs)
    if missing_inputs:
        issues.append("manuscript does not input generated tables: " + ", ".join(missing_inputs))

    abstract_match = re.search(r"\\abstract\{(.*?)\}\s*\\keyword", manuscript, re.DOTALL)
    abstract_words = None
    if abstract_match:
        abstract_words = len(re.findall(r"\b[\w'-]+\b", abstract_match.group(1)))
        if abstract_words > 200:
            issues.append(f"abstract exceeds 200 words: {abstract_words}")
    elif manuscript:
        issues.append("abstract/keyword boundary could not be parsed")

    build_junk_patterns = ("*.aux", "*.log", "*.out", "*.bbl", "*.blg", "*.synctex.gz")
    build_junk = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for pattern in build_junk_patterns
        for path in root.rglob(pattern)
    )
    if build_junk:
        issues.append("build by-products must be excluded from final package: " + ", ".join(build_junk))

    inventory = []
    if root.is_dir():
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: str(p).casefold()):
            inventory.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "protocol": "overleaf_submission_source_check_v1",
        "status": "ready" if not issues else "not_ready",
        "ready": not issues,
        "package_root": str(root),
        "expected_paper_results_root": (
            None
            if expected_paper_results_root is None
            else str(expected_paper_results_root.resolve())
        ),
        "checks": {
            "electronics_documentclass": class_ok,
            "abstract_words": abstract_words,
            "todo_macro_invocations": todo_invocations,
            "todo_lines": todo_lines,
            "citation_keys": len(citations),
            "bib_entries": len(bib_keys),
            "missing_bib_entries": missing_bib,
            "undefined_references": undefined_references,
            "duplicate_labels": duplicate_labels,
            "referenced_figures": graphics,
            "missing_figures": missing_figures,
            "generated_table_inputs_missing": missing_inputs,
            "generated_tables": generated,
            "build_byproducts": build_junk,
        },
        "issues": issues,
        "inventory": inventory,
        "archive_created": False,
        "manuscript_modified": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--paper-results-root",
        type=Path,
        default=DEFAULT_PAPER_RESULTS_ROOT,
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_package(args.package_root, args.paper_results_root)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report is not None:
        report = args.report.resolve()
        if report.exists():
            raise FileExistsError(f"refusing to overwrite existing report: {report}")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
