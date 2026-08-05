"""Stdlib-only identity and gate for formal paper-evaluation environments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Mapping


PROTOCOL = "formal_locked_python_environment_v1"
EXPECTED_PYTHON_VERSION = "3.11.15"
PINNED_REQUIREMENTS_TRAINING_LOCK_SHA256 = (
    "b3e2a46c20ad645e4b0ff1b95edc39d22d5e35e9f39e872a6dbb410b25f75035"
)
_FORBIDDEN_PYTHON_ENV = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONINSPECT",
    "PYTHONWARNINGS",
    "PYTHONBREAKPOINT",
)
_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _locked_requirements(path: Path) -> dict[str, dict[str, str]]:
    path = Path(path).resolve()
    actual_hash = _sha256_file(path)
    if actual_hash != PINNED_REQUIREMENTS_TRAINING_LOCK_SHA256:
        raise RuntimeError(
            "formal requirements lock drifted: "
            f"{actual_hash} != {PINNED_REQUIREMENTS_TRAINING_LOCK_SHA256}"
        )
    locked: dict[str, dict[str, str]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = _REQUIREMENT.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"{path}:{line_number}: requirement is not exactly pinned"
            )
        display_name = match.group("name")
        normalized = _normalized_distribution_name(display_name)
        if normalized in locked:
            raise RuntimeError(
                f"{path}:{line_number}: duplicate locked distribution "
                f"{display_name!r}"
            )
        locked[normalized] = {
            "name": display_name,
            "version": match.group("version"),
        }
    if not locked:
        raise RuntimeError("formal requirements lock is empty")
    return locked


def _installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = _normalized_distribution_name(str(name))
        version = str(distribution.version)
        previous = installed.get(normalized)
        if previous is not None and previous != version:
            raise RuntimeError(
                f"multiple installed versions for {normalized}: "
                f"{previous!r} and {version!r}"
            )
        installed[normalized] = version
    return installed


def validate_locked_distributions(
    locked: Mapping[str, Mapping[str, str]],
    installed: Mapping[str, str],
) -> dict[str, str]:
    """Return exact locked versions or fail on a missing/version-drifted pin."""

    verified: dict[str, str] = {}
    for normalized, record in sorted(locked.items()):
        expected = str(record["version"])
        actual = installed.get(normalized)
        if actual != expected:
            raise RuntimeError(
                "formal installed distribution drifted for "
                f"{record['name']}: expected {expected!r}, got {actual!r}"
            )
        verified[str(record["name"])] = actual
    return verified


def formal_environment_identity(
    project_dir: Path,
    *,
    validate_process: bool,
) -> dict[str, Any]:
    """Validate the formal interpreter/lock and return its exact identity."""

    project_dir = Path(project_dir).resolve()
    expected_python = (
        project_dir / ".venv" / "Scripts" / "python.exe"
    ).resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(
            "formal evaluation requires the project .venv interpreter: "
            f"{expected_python}; got {actual_python}"
        )
    actual_python_version = platform.python_version()
    if actual_python_version != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            "formal Python version drifted: expected "
            f"{EXPECTED_PYTHON_VERSION}, got {actual_python_version}"
        )
    if validate_process and (
        int(sys.flags.safe_path) != 1
        or int(sys.flags.no_user_site) != 1
        or int(sys.flags.ignore_environment) != 0
        or os.environ.get("PYTHONHASHSEED") != "20260722"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or any(name in os.environ for name in _FORBIDDEN_PYTHON_ENV)
    ):
        raise RuntimeError(
            "formal evaluation requires -P -s, the frozen hash/CUBLAS "
            "environment, and no Python path/startup overrides"
        )
    lock_path = (
        project_dir / "experiments" / "requirements-training.lock.txt"
    ).resolve()
    locked = _locked_requirements(lock_path)
    verified = validate_locked_distributions(
        locked,
        _installed_distributions(),
    )
    return {
        "protocol": PROTOCOL,
        "python_executable": str(actual_python),
        "python_version": actual_python_version,
        "python_build": sys.version,
        "python_implementation": platform.python_implementation(),
        "requirements_lock_path": str(lock_path),
        "requirements_lock_sha256": (
            PINNED_REQUIREMENTS_TRAINING_LOCK_SHA256
        ),
        "locked_distributions": verified,
    }


__all__ = [
    "EXPECTED_PYTHON_VERSION",
    "PINNED_REQUIREMENTS_TRAINING_LOCK_SHA256",
    "PROTOCOL",
    "formal_environment_identity",
    "validate_locked_distributions",
]
