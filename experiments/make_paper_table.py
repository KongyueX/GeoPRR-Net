"""Merge frozen SyncG-test and external-test metrics into one Markdown table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = (
    "Original Transformer",
    "Geometry-v1",
    "Geometry-v2",
    "Mean Fusion",
    "Quality-weighted Fusion",
    "Residual without Gate",
    "Ours",
)


def _load_metrics(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} does not contain a metrics object")
    return metrics


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syncg", type=Path, required=True, help="SyncG test metrics.json")
    external = parser.add_mutually_exclusive_group(required=True)
    external.add_argument(
        "--external",
        type=Path,
        help="frozen external-test metrics.json",
    )
    # Preserve old invocations while making the table dataset-agnostic.
    external.add_argument(
        "--realgauges",
        dest="external",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--external-name", default="RPM-10K single-pointer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    syncg = _load_metrics(args.syncg)
    external_metrics = _load_metrics(args.external)
    external_name = args.external_name.replace("|", "/")
    lines = [
        "| Method | SyncG E2E NMAE ↓ | SyncG Acc@2% ↑ | SyncG coverage ↑ | "
        f"{external_name} E2E NMAE ↓ | {external_name} Accε@1% ↑ | "
        f"{external_name} Accθ@5% ↑ | {external_name} coverage ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        syncg_row = syncg.get(method, {})
        external_row = external_metrics.get(method, {})
        lines.append(
            "| "
            + " | ".join(
                (
                    method,
                    _fmt(syncg_row.get("nmae")),
                    _fmt(syncg_row.get("acc_2pct")),
                    _fmt(syncg_row.get("coverage")),
                    _fmt(external_row.get("nmae")),
                    _fmt(external_row.get("dialbench_acc_epsilon_e2e")),
                    _fmt(external_row.get("dialbench_acc_theta_e2e")),
                    _fmt(external_row.get("coverage")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "_Normalized headline metrics use the full evaluable set: a missing "
            "reading receives NMAE penalty 1.0 and counts as incorrect for "
            "accuracy; coverage is reported separately. Accε and Accθ use the "
            "DialBench Ref≤0.01 and Rel<0.05 thresholds, but this known-range "
            "single-pointer subset is not the complete official leaderboard "
            "setting. Uncapped successful-output Ref/Rel remain in metrics.json._",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
