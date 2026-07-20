"""Build a compact SyncG-test feature-ablation table from frozen metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must use NAME=METRICS_JSON")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("variant name/path cannot be empty")
    return name.strip(), Path(path.strip())


def _metrics(path: Path, method: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    if method not in metrics:
        raise ValueError(f"{path} has no {method!r} metrics")
    return metrics[method]


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument(
        "--variant",
        type=_variant,
        action="append",
        default=[],
        metavar="NAME=METRICS_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        ("Quality-weighted (no residual)", _metrics(args.full, "Quality-weighted Fusion")),
        ("Residual without Gate", _metrics(args.full, "Residual without Gate")),
        ("Full Ours", _metrics(args.full, "Ours")),
    ]
    rows.extend(
        (name, _metrics(path, "Ours"))
        for name, path in args.variant
    )
    lines = [
        "| Variant | NMAE ↓ | Acc@2% ↑ | Coverage ↑ | "
        "Correction coverage ↑ | Negative transfer ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    name.replace("|", "/"),
                    _fmt(metrics.get("nmae")),
                    _fmt(metrics.get("acc_2pct")),
                    _fmt(metrics.get("coverage")),
                    _fmt(metrics.get("correction_coverage")),
                    _fmt(metrics.get("negative_transfer_rate")),
                )
            )
            + " |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
