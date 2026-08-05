"""Dependency-neutral strict JSON readers for research artifacts.

The helpers in this module deliberately sit upstream of both PEPD and FADR.
They reject Python's permissive JSON extensions (NaN and infinities), duplicate
object keys, non-object top-level values, and empty JSONL inputs.  Research
pipelines can therefore bind this file's source hash without depending on a
downstream experiment protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


STRICT_JSON_PROTOCOL = "research_artifact_strict_json_v1"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, label=f"{label}[{index}]")


def strict_json_load(path: Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate keys/non-finite numbers."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{resolved} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} is not a JSON object")
    _reject_nonfinite(value, label=str(resolved))
    return value


def strict_jsonl_load(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL under the same strict object semantics."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_constant,
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{resolved}:{line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{resolved}:{line_number} is not a JSON object"
                )
            _reject_nonfinite(row, label=f"{resolved}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{resolved} contains no JSONL rows")
    return rows


def strict_json_source_sha256() -> str:
    """Return the byte-level source identity of this strict-reader module."""

    digest = hashlib.sha256()
    with Path(__file__).resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
