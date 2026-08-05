"""Build publication figures from immutable aggregate experiment summaries.

The script reads aggregate JSON only.  It never opens field images, labels, or
per-sample predictions and performs no model inference or statistical fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELD = (
    ROOT
    / "artifacts/runs/field_holdout_xiangmu2_2026/physical_entity_leakage_sensitivity_v1/summary.json"
)
DEFAULT_OOF = ROOT / "artifacts/runs/vdn_official200_train_oof_v1/comparison.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") not in {"complete", "frozen"}:
        raise ValueError(f"not a complete aggregate artifact: {path}")
    return value


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def field_comparison(field: dict[str, Any], output_dir: Path) -> None:
    methods = field["sensitivity_metrics"]
    values = {
        "Original\nTransformer": methods["original_transformer"],
        "Geometric\nmask": methods["base_mask"],
        "VDN": methods["vdn_official200"],
        "PEPD+\nFADR": methods["pepd_fadr"],
        "PEPD": methods["pepd_only"],
    }
    labels = list(values)
    nmae = [float(values[label]["full_denominator_nmae"]) for label in labels]
    acc5 = [float(values[label]["acc_at_5pct"]) for label in labels]
    colors = ["#8c8c8c", "#8c8c8c", "#5b6f8f", "#d88b37", "#276fbf"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    x = np.arange(len(labels))
    bars = axes[0].bar(x, nmae, color=colors, width=0.72)
    axes[0].set_ylabel("Full-denominator NMAE ↓")
    axes[0].set_xticks(x, labels)
    axes[0].tick_params(axis="x", labelsize=6.2)
    axes[0].set_ylim(0.0, max(nmae) * 1.18)
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.55)
    axes[0].set_axisbelow(True)
    for bar, value in zip(bars, nmae, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(nmae) * 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    bars = axes[1].bar(x, np.asarray(acc5) * 100.0, color=colors, width=0.72)
    axes[1].set_ylabel("Acc@5% (%) ↑")
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(axis="x", labelsize=6.2)
    axes[1].set_ylim(0.0, 108.0)
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.55)
    axes[1].set_axisbelow(True)
    for bar, value in zip(bars, acc5, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{value * 100:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    fig.subplots_adjust(wspace=0.30)
    save(fig, output_dir, "field-main-comparison")


def field_effects(
    field: dict[str, Any],
    output_dir: Path,
) -> None:
    effects = field["paired_effects"]

    rows = [
        (
            "Original Transformer",
            float(
                effects["original_transformer_minus_pepd_only"][
                    "delta_nmae_comparator_minus_pepd"
                ]
            ),
            list(
                effects["original_transformer_minus_pepd_only"][
                    "paired_physical_group_bootstrap_95ci"
                ]
            ),
        ),
        (
            "VDN official-200",
            float(
                effects["vdn_official200_minus_pepd_only"][
                    "delta_nmae_comparator_minus_pepd"
                ]
            ),
            list(
                effects["vdn_official200_minus_pepd_only"][
                    "paired_physical_group_bootstrap_95ci"
                ]
            ),
        ),
        (
            "Base mask",
            float(
                effects["base_mask_minus_pepd_only"][
                    "delta_nmae_comparator_minus_pepd"
                ]
            ),
            list(
                effects["base_mask_minus_pepd_only"][
                    "paired_physical_group_bootstrap_95ci"
                ]
            ),
        ),
        (
            "PEPD+FADR",
            float(
                effects["pepd_fadr_minus_pepd_only"][
                    "delta_nmae_comparator_minus_pepd"
                ]
            ),
            list(
                effects["pepd_fadr_minus_pepd_only"][
                    "paired_physical_group_bootstrap_95ci"
                ]
            ),
        ),
    ]
    fig, ax = plt.subplots(figsize=(5.35, 2.55))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0.0, color="#4d4d4d", linewidth=0.8)
    for ypos, (label, point, interval) in zip(y, rows, strict=True):
        lower, upper = interval
        ax.errorbar(
            point,
            ypos,
            xerr=np.asarray([[point - lower], [upper - point]]),
            fmt="o",
            markersize=5.5,
            markerfacecolor="#276fbf",
            markeredgecolor="#276fbf",
            ecolor="#276fbf",
            elinewidth=1.1,
            capsize=2.5,
        )
        ax.text(upper + 0.007, ypos, f"{point:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlabel("Comparator NMAE − PEPD NMAE (positive favors PEPD)")
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlim(-0.015, max(row[2][1] for row in rows) + 0.055)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.plot([], [], "o", color="#276fbf", label="Post-hoc leakage sensitivity; 95% group bootstrap")
    ax.legend(loc="lower right", frameon=False)
    save(fig, output_dir, "field-paired-effects")


def fadr_transfer(field: dict[str, Any], oof: dict[str, Any], output_dir: Path) -> None:
    train_pepd = 0.14849714937173064
    train_fadr = float(oof["metrics"]["fadr_v2_full"]["three_seed_mean"]["nmae"])
    field_pepd = float(
        field["sensitivity_metrics"]["pepd_only"]["full_denominator_nmae"]
    )
    field_fadr = float(
        field["sensitivity_metrics"]["pepd_fadr"]["full_denominator_nmae"]
    )
    data = np.asarray([[train_pepd, train_fadr], [field_pepd, field_fadr]])
    domains = [
        "SyncG train grouped OOF\n(4,380 images / 197 groups)",
        "Field leakage sensitivity\n(725 images / 18 groups)",
    ]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.3, 2.75))
    bars_pepd = ax.bar(x - width / 2, data[:, 0], width, label="PEPD", color="#276fbf")
    bars_fadr = ax.bar(x + width / 2, data[:, 1], width, label="PEPD+FADR", color="#d88b37")
    ax.set_ylabel("Full-denominator NMAE ↓")
    ax.set_xticks(x, domains)
    ax.set_ylim(0.0, 0.175)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    for bars in (bars_pepd, bars_fadr):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    train_change = (train_fadr - train_pepd) / train_pepd * 100.0
    field_change = (field_fadr - field_pepd) / field_pepd * 100.0
    ax.text(x[0], 0.163, f"FADR change: {train_change:.1f}%", ha="center", fontsize=7)
    ax.text(x[1], 0.090, f"FADR change: +{field_change:.1f}%", ha="center", fontsize=7)
    save(fig, output_dir, "fadr-domain-transfer")


def main() -> None:
    args = parse_args()
    configure_style()
    field = load(args.field.resolve())
    oof = load(args.oof.resolve())
    output_dir = args.output_dir.resolve()
    field_comparison(field, output_dir)
    field_effects(field, output_dir)
    fadr_transfer(field, oof, output_dir)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "figures": 3}))


if __name__ == "__main__":
    main()
