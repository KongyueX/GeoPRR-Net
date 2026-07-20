"""Summarize end-to-end front-end failures from frozen prediction caches."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FAILURE_ORDER = (
    "meter_not_found",
    "pointer_not_found",
    "reading_backend_failed",
    "collector_exception",
    "other_failure",
)


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=predictions.jsonl")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected NAME=predictions.jsonl")
    return name, path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _method_succeeded(row: dict[str, Any], method: str) -> bool:
    payload = (row.get("methods") or {}).get(method) or {}
    return bool(payload.get("status") and payload.get("prediction") is not None)


def _failure_category(row: dict[str, Any]) -> str:
    code = str(row.get("error_code") or "").strip()
    if code in FAILURE_ORDER:
        return code
    if code.startswith("collector_"):
        return "collector_exception"
    if code:
        return "other_failure"
    return "other_failure"


def summarize_cache(name: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} contains no rows")

    failures: Counter[str] = Counter()
    weighted_success = 0
    v1_success = 0
    v2_success = 0
    both_geometry_success = 0
    for row in rows:
        weighted_ok = _method_succeeded(row, "weighted_fusion")
        v1_ok = _method_succeeded(row, "geometry_v1")
        v2_ok = _method_succeeded(row, "geometry_v2")
        weighted_success += int(weighted_ok)
        v1_success += int(v1_ok)
        v2_success += int(v2_ok)
        both_geometry_success += int(v1_ok and v2_ok)
        if not weighted_ok:
            failures[_failure_category(row)] += 1

    total = len(rows)
    return {
        "name": name,
        "source": str(path.resolve()),
        "samples": total,
        "weighted_success": weighted_success,
        "weighted_coverage": weighted_success / total,
        "geometry_v1_success": v1_success,
        "geometry_v2_success": v2_success,
        "both_geometry_success": both_geometry_success,
        "failures": {
            category: int(failures.get(category, 0))
            for category in FAILURE_ORDER
        },
    }


def _count_percent(count: int, total: int) -> str:
    return f"{count} ({100.0 * count / max(1, total):.1f}%)"


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# End-to-end failure attribution",
        "",
        (
            "Counts use every frozen manifest row. A failed prediction is never "
            "removed from the denominator."
        ),
        "",
        (
            "| Dataset | N | Success | Meter not found | Pointer not found | "
            "Reading backend | Collector | Other |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        total = int(summary["samples"])
        failures = summary["failures"]
        lines.append(
            "| {name} | {total} | {success} | {meter} | {pointer} | "
            "{reading} | {collector} | {other} |".format(
                name=summary["name"],
                total=total,
                success=_count_percent(
                    int(summary["weighted_success"]),
                    total,
                ),
                meter=_count_percent(
                    int(failures["meter_not_found"]),
                    total,
                ),
                pointer=_count_percent(
                    int(failures["pointer_not_found"]),
                    total,
                ),
                reading=_count_percent(
                    int(failures["reading_backend_failed"]),
                    total,
                ),
                collector=_count_percent(
                    int(failures["collector_exception"]),
                    total,
                ),
                other=_count_percent(
                    int(failures["other_failure"]),
                    total,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Geometry estimator availability",
            "",
            "| Dataset | Geometry-v1 | Geometry-v2 | Both |",
            "|---|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        total = int(summary["samples"])
        lines.append(
            "| {name} | {v1} | {v2} | {both} |".format(
                name=summary["name"],
                v1=_count_percent(
                    int(summary["geometry_v1_success"]),
                    total,
                ),
                v2=_count_percent(
                    int(summary["geometry_v2_success"]),
                    total,
                ),
                both=_count_percent(
                    int(summary["both_geometry_success"]),
                    total,
                ),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_named_path,
        required=True,
        metavar="NAME=PREDICTIONS",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [name for name, _ in args.dataset]
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    summaries = [
        summarize_cache(name, path)
        for name, path in args.dataset
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summaries), encoding="utf-8")
    json_path = args.output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "protocol": "frozen_end_to_end_failure_attribution_v1",
                "datasets": summaries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    print(json_path.resolve())


if __name__ == "__main__":
    main()
