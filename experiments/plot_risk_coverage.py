"""Plot frozen risk-coverage diagnostics from evaluation CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must use NAME=CSV")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("dataset name/path cannot be empty")
    return name.strip(), Path(raw_path.strip())


def _read_rows(path: Path) -> list[dict[str, float | None]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, float | None]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | None] = {}
            for key in (
                "threshold",
                "correction_coverage",
                "selected_nmae",
                "system_nmae",
                "negative_transfer_rate",
            ):
                value: Any = raw.get(key)
                row[key] = (
                    None
                    if value in (None, "", "None")
                    else float(value)
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no risk-coverage rows")
    return rows


def _xy(
    rows: list[dict[str, float | None]],
    y_key: str,
) -> tuple[list[float], list[float]]:
    points = [
        (row["correction_coverage"], row[y_key])
        for row in rows
        if row["correction_coverage"] is not None and row[y_key] is not None
    ]
    points.sort(key=lambda value: float(value[0]))
    return (
        [float(point[0]) for point in points],
        [float(point[1]) for point in points],
    )


def plot_risk_coverage(
    datasets: list[tuple[str, Path]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for name, path in datasets:
        rows = _read_rows(path)
        coverage, system_nmae = _xy(rows, "system_nmae")
        axes[0].plot(
            coverage,
            system_nmae,
            linewidth=2.0,
            label=name,
        )
        coverage, negative_transfer = _xy(
            rows,
            "negative_transfer_rate",
        )
        axes[1].plot(
            coverage,
            negative_transfer,
            linewidth=2.0,
            label=name,
        )

    axes[0].set_title("System risk vs correction coverage")
    axes[0].set_xlabel("Correction coverage")
    axes[0].set_ylabel("End-to-end NMAE")
    axes[1].set_title("Negative transfer vs correction coverage")
    axes[1].set_xlabel("Correction coverage")
    axes[1].set_ylabel("Negative transfer rate")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.set_xlim(0.0, 1.0)
        axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_dataset_argument,
        required=True,
        metavar="NAME=CSV",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.dataset]
    if len(set(names)) != len(names):
        raise ValueError("dataset names must be unique")
    plot_risk_coverage(args.dataset, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
