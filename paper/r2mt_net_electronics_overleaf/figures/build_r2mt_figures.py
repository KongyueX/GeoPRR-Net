"""Build R2MT-Net manuscript figures and compact source-data tables.

The script consumes completed repository artifacts.  OCR is retained as the
end-to-end range branch, but its paper-facing numerical result is deliberately
limited to accuracy conditional on the labeled frames accepted by that branch.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "artifacts" / "runs"
PROBES = ROOT / "artifacts" / "probes"
DOWNSTREAM = RUNS / "r2mt_downstream_repro_20260825"
RESULTS = ROOT / "results"

CONDITIONS = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
PROJECTIVE = frozenset(CONDITIONS[3:])
METRICS = ("all_conditions", "projective_pooled")
SEEDS = (20262020, 20262021, 20262022)

CONDITION_LABELS = {
    "clean": "Clean",
    "blur_moderate": "Blur-M",
    "blur_severe": "Blur-S",
    "perspective_moderate": "Persp.-25",
    "perspective_severe": "Persp.-45",
    "combined_severe": "45 + blur",
    "all_conditions": "All six",
    "projective_pooled": "Projective pool",
}

COLORS = {
    "ink": "#1F2933",
    "muted": "#66788A",
    "grid": "#D9E1E8",
    "blue": "#135BA1",
    "blue_soft": "#DCEAF7",
    "teal": "#198F8A",
    "teal_soft": "#DDF2F0",
    "orange": "#D46A1F",
    "orange_soft": "#F8E6D8",
    "purple": "#6B5AA6",
    "purple_soft": "#E9E5F5",
    "rose": "#B3445A",
    "rose_soft": "#F5DFE4",
    "gray": "#7C8792",
    "gray_soft": "#EEF1F4",
}

METHOD_COLORS = {
    "Direct-ResNet18": "#7C8792",
    "EfficientNet-B0": "#C58D36",
    "MobileNetV3-Large": "#9B7FB5",
    "R2MT-Net": "#135BA1",
}

CONDITION_COLORS = {
    "clean": "#AAB2BA",
    "blur_moderate": "#8796A3",
    "blur_severe": "#667886",
    "perspective_moderate": "#69B9B2",
    "perspective_severe": "#238F8C",
    "combined_severe": "#0C6674",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def save_all(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        HERE / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.03,
        1.03,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["ink"],
    )


def schematic_axis(ax: plt.Axes, title: str, label: str) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    panel_label(ax, label)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    face: str = "white",
    edge: str = COLORS["gray"],
    fontsize: float = 7.2,
    weight: str = "normal",
    radius: float = 0.018,
) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
        linespacing=1.05,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def load_main_rows() -> list[dict[str, Any]]:
    ablation = read_json(PROBES / "r2mt_ablation_summary.json")
    full = next(row for row in ablation["rows"] if row["method"] == "R2MT-Net (full)")
    reference = read_json(
        RUNS
        / "r2mt_final_conservative"
        / f"seed_{SEEDS[0]}"
        / "six_condition_results.json"
    )
    base = read_json(
        RUNS
        / "remst_resnet18_raw_context_correction_adapt"
        / "three_seed_main_table_summary.json"
    )["main_table"]

    rows: list[dict[str, Any]] = []
    external_names = {
        "Direct-ResNet18": "SARN-v2+Direct-ResNet18",
        "EfficientNet-B0": "SARN-v2+EfficientNet-B0",
        "MobileNetV3-Large": "SARN-v2+MobileNetV3-Large",
    }
    for metric in (*CONDITIONS, *METRICS):
        for display, source_name in external_names.items():
            payload = reference["summary"][metric]["external_models"][source_name]
            per_seed = [float(item["nmae"]) for item in payload["per_seed"]]
            rows.append(
                {
                    "metric": metric,
                    "metric_label": CONDITION_LABELS[metric],
                    "method": display,
                    "mean_nmae": float(np.mean(per_seed)),
                    "sample_sd": float(np.std(per_seed, ddof=1)),
                    "n_images": 1558,
                    "n_scene_groups": 14,
                    "n_seeds": 3,
                    "evidence_status": "completed matched external ledger",
                }
            )

        base_payload = base[metric]
        rows.append(
            {
                "metric": metric,
                "metric_label": CONDITION_LABELS[metric],
                "method": "ReMST base",
                "mean_nmae": float(base_payload["candidate_mean_nmae"]),
                "sample_sd": float(base_payload["candidate_sample_std_nmae"]),
                "n_images": 1558,
                "n_scene_groups": 14,
                "n_seeds": 3,
                "evidence_status": "completed internal base",
            }
        )
        full_payload = full["metrics"][metric]
        rows.append(
            {
                "metric": metric,
                "metric_label": CONDITION_LABELS[metric],
                "method": "R2MT-Net",
                "mean_nmae": float(full_payload["mean_nmae"]),
                "sample_sd": float(full_payload["sample_sd"]),
                "n_images": 1558,
                "n_scene_groups": 14,
                "n_seeds": 3,
                "evidence_status": "completed retrospective development evaluation",
            }
        )
    write_csv(
        HERE / "source_r2mt_main.csv",
        rows,
        [
            "metric",
            "metric_label",
            "method",
            "mean_nmae",
            "sample_sd",
            "n_images",
            "n_scene_groups",
            "n_seeds",
            "evidence_status",
        ],
    )
    return rows


def load_ablation_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = read_json(PROBES / "r2mt_ablation_summary.json")
    base = read_json(
        RUNS
        / "remst_resnet18_raw_context_correction_adapt"
        / "three_seed_main_table_summary.json"
    )["main_table"]
    rows: list[dict[str, Any]] = []
    for metric in (*CONDITIONS, *METRICS):
        rows.append(
            {
                "method": "ReMST base",
                "metric": metric,
                "metric_label": CONDITION_LABELS[metric],
                "mean_nmae": float(base[metric]["candidate_mean_nmae"]),
                "sample_sd": float(base[metric]["candidate_sample_std_nmae"]),
                "source_type": "three_seed_evaluation",
            }
        )
    for row in summary["rows"]:
        for metric in (*CONDITIONS, *METRICS):
            payload = row["metrics"][metric]
            rows.append(
                {
                    "method": row["method"].replace("R2MT-Net (full)", "R2MT-Net"),
                    "metric": metric,
                    "metric_label": CONDITION_LABELS[metric],
                    "mean_nmae": float(payload["mean_nmae"]),
                    "sample_sd": float(payload["sample_sd"]),
                    "source_type": row["source_type"],
                }
            )
    write_csv(
        HERE / "source_r2mt_ablation.csv",
        rows,
        ["method", "metric", "metric_label", "mean_nmae", "sample_sd", "source_type"],
    )

    paired = read_json(PROBES / "r2mt_paired_statistics.json")
    effects: list[dict[str, Any]] = []

    def records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
        payload = read_json(path)
        return {
            (str(item["sample_id"]), str(item["condition"])): {
                "scene": str(item["scene_stem"]),
                "target": float(item["normalized_target"]),
                "prediction": float(item["mett"]["prediction"]),
            }
            for item in payload["per_sample_condition"]
        }

    full_runs = [
        records(RUNS / "r2mt_final_conservative" / f"seed_{seed}" / "six_condition_results.json")
        for seed in SEEDS
    ]
    base_runs = [
        records(
            RUNS
            / "remst_resnet18_raw_context_correction_adapt"
            / f"seed_{seed}"
            / "six_condition_results.json"
        )
        for seed in SEEDS
    ]

    def paired_base_effect(metric: str) -> dict[str, float]:
        selected = (
            frozenset(CONDITIONS)
            if metric == "all_conditions"
            else (PROJECTIVE if metric == "projective_pooled" else frozenset((metric,)))
        )
        scene_sums: dict[str, float] = {}
        scene_counts: dict[str, int] = {}
        for full_run, base_run in zip(full_runs, base_runs):
            if full_run.keys() != base_run.keys():
                raise ValueError("R2MT and ReMST base paired rows differ")
            for key in sorted(full_run):
                if key[1] not in selected:
                    continue
                full_item = full_run[key]
                base_item = base_run[key]
                if full_item["scene"] != base_item["scene"] or abs(full_item["target"] - base_item["target"]) > 1.0e-12:
                    raise ValueError("R2MT and ReMST base paired metadata differ")
                difference = abs(base_item["prediction"] - base_item["target"]) - abs(full_item["prediction"] - full_item["target"])
                scene = full_item["scene"]
                scene_sums[scene] = scene_sums.get(scene, 0.0) + difference
                scene_counts[scene] = scene_counts.get(scene, 0) + 1
        scenes = sorted(scene_sums)
        sums = np.asarray([scene_sums[scene] for scene in scenes], dtype=np.float64)
        counts = np.asarray([scene_counts[scene] for scene in scenes], dtype=np.float64)
        # The random generator resamples observed scene-cluster sums for the
        # reported nonparametric bootstrap; it does not create simulated data.
        metric_seed = 20262031 + (*CONDITIONS, *METRICS).index(metric)
        rng = np.random.Generator(np.random.PCG64(metric_seed))
        draws = rng.integers(0, len(scenes), size=(20_000, len(scenes)))
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        return {
            "estimate": float(sums.sum() / counts.sum()),
            "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975)),
        }

    for metric in (*CONDITIONS, *METRICS):
        effect = paired_base_effect(metric)
        effects.append(
            {
                "comparator": "ReMST base",
                "metric": metric,
                "metric_label": CONDITION_LABELS[metric],
                "reduction_pp": 100.0 * effect["estimate"],
                "ci_low_pp": 100.0 * effect["low"],
                "ci_high_pp": 100.0 * effect["high"],
                "bootstrap_replicates": 20_000,
                "bootstrap_unit": "scene_stem",
                "formal_holdout_access": False,
            }
        )
    for comparator, metrics in paired["comparisons"].items():
        for metric in (*CONDITIONS, *METRICS):
            payload = metrics[metric]
            effects.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "metric_label": CONDITION_LABELS[metric],
                    "reduction_pp": 100.0 * float(payload["nmae_reduction_vs_comparator"]),
                    "ci_low_pp": 100.0 * float(payload["ci95_scene_cluster"][0]),
                    "ci_high_pp": 100.0 * float(payload["ci95_scene_cluster"][1]),
                    "bootstrap_replicates": int(paired["bootstrap_replicates"]),
                    "bootstrap_unit": paired["bootstrap_unit"],
                    "formal_holdout_access": paired["formal_holdout_access"],
                }
            )
    write_csv(
        HERE / "source_r2mt_paired_effects.csv",
        effects,
        [
            "comparator",
            "metric",
            "metric_label",
            "reduction_pp",
            "ci_low_pp",
            "ci_high_pp",
            "bootstrap_replicates",
            "bootstrap_unit",
            "formal_holdout_access",
        ],
    )
    return rows, effects


def load_routing_rows() -> list[dict[str, Any]]:
    payload = read_json(PROBES / "r2mt_conservative_routing_stability.json")
    rows: list[dict[str, Any]] = []
    sources = {
        **{metric: payload["conditions"][metric] for metric in CONDITIONS},
        "all_conditions": payload["all_conditions"],
        "projective_pooled": payload["projective_pooled"],
    }
    for metric, source in sources.items():
        for route in ("fixed", "adaptive", "conservative"):
            rows.append(
                {
                    "route": route,
                    "metric": metric,
                    "metric_label": CONDITION_LABELS[metric],
                    "mean_nmae": float(source[route]["mean_nmae"]),
                    "sample_sd": float(source[route]["sample_sd"]),
                    "adaptive_strength": 0.0 if route == "fixed" else (1.0 if route == "adaptive" else float(payload["adaptive_strength"])),
                    "status": payload["status"],
                    "formal_holdout_access": payload["formal_holdout_access"],
                }
            )
    write_csv(
        HERE / "source_r2mt_routing_sensitivity.csv",
        rows,
        [
            "route",
            "metric",
            "metric_label",
            "mean_nmae",
            "sample_sd",
            "adaptive_strength",
            "status",
            "formal_holdout_access",
        ],
    )
    return rows


def require_completed(payload: dict[str, Any], protocol: str) -> None:
    """Reject accidentally selected partial or differently scoped artifacts."""
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete artifact for {protocol}: {payload.get('status')!r}")
    if payload.get("protocol") != protocol:
        raise ValueError(
            f"Unexpected protocol: expected {protocol!r}, got {payload.get('protocol')!r}"
        )


def load_field_transfer_rows() -> list[dict[str, Any]]:
    payload = read_json(RESULTS / "r2mt_industrial_multimethod.json")
    require_completed(payload, "r2mt_industrial_multimethod_summary_v1")
    cohort = payload["cohort"]
    if int(cohort["samples"]) != 1395 or int(cohort["physical_groups"]) != 52:
        raise ValueError("Unexpected industrial cohort size")
    if payload["methods"] != [
        "R²MT-Net",
        "Direct-ResNet18",
        "EfficientNet-B0",
        "MobileNetV3-Large",
    ]:
        raise ValueError("Unexpected industrial method roster")

    rows: list[dict[str, Any]] = []
    for metric in (*CONDITIONS, *METRICS):
        subset = payload["metrics"][metric]
        for method in payload["methods"]:
            source = subset["methods"][method]
            paired = (
                None
                if method == "R²MT-Net"
                else payload["paired_effects"][method][metric]
            )
            ci = (
                [0.0, 0.0]
                if paired is None
                else paired["ci95_group_cluster_percentage_points"]
            )
            rows.append(
                {
                    "metric": metric,
                    "metric_label": CONDITION_LABELS[metric],
                    "risk_group": (
                        "aggregate"
                        if metric in METRICS
                        else (
                            "projective"
                            if metric in PROJECTIVE
                            else "non-projective"
                        )
                    ),
                    "method": method,
                    "mean_nmae": float(source["mean_nmae"]),
                    "sample_sd": float(source["sample_sd"]),
                    "mean_coverage": float(source["mean_coverage"]),
                    "r2mt_relative_reduction_percent": float(
                        source["r2mt_relative_reduction_percent"]
                    ),
                    "r2mt_paired_reduction_pp": (
                        ""
                        if paired is None
                        else float(paired["nmae_reduction_percentage_points"])
                    ),
                    "ci95_low_pp": "" if paired is None else float(ci[0]),
                    "ci95_high_pp": "" if paired is None else float(ci[1]),
                    "ci95_excludes_zero": (
                        ""
                        if paired is None
                        else bool(float(ci[0]) > 0.0 or float(ci[1]) < 0.0)
                    ),
                    "bootstrap_probability_positive": (
                        ""
                        if paired is None
                        else float(paired["bootstrap_probability_positive"])
                    ),
                    "rows_per_seed": int(subset["rows_per_seed"]),
                    "source_samples": int(cohort["samples"]),
                    "source_groups": int(cohort["physical_groups"]),
                    "n_seeds": len(payload["seeds"]),
                    "bootstrap_replicates": int(payload["bootstrap"]["replicates"]),
                    "bootstrap_unit": payload["bootstrap"]["unit"],
                }
            )
    write_csv(
        HERE / "source_r2mt_industrial_multimethod.csv",
        rows,
        list(rows[0]),
    )
    return rows


def load_vdn_rows() -> list[dict[str, Any]]:
    payload = read_json(DOWNSTREAM / "vdn" / "intersection_summary.json")
    require_completed(payload, "r2mt_vdn_double_holdout_intersection_v1")
    cohort = payload["cohort"]
    if int(cohort["intersection_samples"]) != 129 or int(cohort["intersection_rows"]) != 774:
        raise ValueError("Unexpected VDN intersection size")

    rows: list[dict[str, Any]] = []
    for metric in (*CONDITIONS, *METRICS):
        source = payload["summary"][metric]
        candidate = source["remstnet_v3"]["metric_across_seed_mean_sd"]["nmae"]
        efficient = source["sarn_v2_efficientnet_b0"]["metric_across_seed_mean_sd"]["nmae"]
        annotation = source["vdn_official200_annotation_reference"]
        effect_eff = source["paired_scene_bootstrap"]["versus_sarn_v2_efficientnet_b0"]
        effect_vdn = source["paired_scene_bootstrap"]["versus_vdn_annotation_reference"]
        rows.append(
            {
                "metric": metric,
                "metric_label": CONDITION_LABELS[metric],
                "rows": int(source["rows"]),
                "samples": int(source["samples"]),
                "scene_groups": int(source["scene_groups"]),
                "r2mt_mean_nmae": float(candidate["mean"]),
                "r2mt_sample_sd": float(candidate["sample_sd"]),
                "efficientnet_mean_nmae": float(efficient["mean"]),
                "efficientnet_sample_sd": float(efficient["sample_sd"]),
                "vdn_annotation_nmae": float(annotation["metrics"]["nmae"]),
                "vdn_annotation_single_checkpoint": bool(annotation["single_checkpoint"]),
                "vdn_annotation_deployable": bool(annotation["deployable"]),
                "r2mt_reduction_vs_efficientnet_pp": -100.0 * float(effect_eff["delta_nmae"]),
                "r2mt_reduction_vs_efficientnet_ci_low_pp": -100.0
                * float(effect_eff["scene_grouped_bootstrap_ci95"]["high"]),
                "r2mt_reduction_vs_efficientnet_ci_high_pp": -100.0
                * float(effect_eff["scene_grouped_bootstrap_ci95"]["low"]),
                "r2mt_reduction_vs_vdn_pp": -100.0 * float(effect_vdn["delta_nmae"]),
                "r2mt_reduction_vs_vdn_ci_low_pp": -100.0
                * float(effect_vdn["scene_grouped_bootstrap_ci95"]["high"]),
                "r2mt_reduction_vs_vdn_ci_high_pp": -100.0
                * float(effect_vdn["scene_grouped_bootstrap_ci95"]["low"]),
            }
        )
    write_csv(
        HERE / "source_r2mt_vdn.csv",
        rows,
        list(rows[0]),
    )
    return rows


def load_ocr_accepted_rows() -> list[dict[str, Any]]:
    payload = read_json(DOWNSTREAM / "ocr" / "score.json")
    require_completed(payload, "r2mt_ocr_end_to_end_deployment_repro_v1_score")
    labeled = int(payload["samples"]["labeled_full_frames_for_accuracy"])
    metric = payload["operating_point_metrics"]["decoder_default"]["across_seed_mean_sd"]
    accepted = int(payload["range_metrics_on_labeled_frames"]["decoder_passed"])
    rows = [
        {
            "reporting_scope": "accepted labeled frames only",
            "accepted_labeled_frames": accepted,
            "all_labeled_frames": labeled,
            "conditional_accuracy_at_5_percent": float(metric["acc_at_5_percent_conditional"]["mean"]),
            "conditional_accuracy_at_5_percent_sample_sd": float(metric["acc_at_5_percent_conditional"]["sample_sd"]),
            "n_seeds": 3,
            "detector_boxes": "cached; live detector recall and latency excluded",
        }
    ]
    write_csv(
        HERE / "source_r2mt_ocr_accepted.csv",
        rows,
        list(rows[0]),
    )
    return rows


def load_efficiency_rows() -> list[dict[str, Any]]:
    arms = {
        "raw_only": "Raw anchor",
        "twin_endpoint": "Dual-view endpoint",
        "full_mett": "R2MT-Net",
    }
    rows: list[dict[str, Any]] = []
    for arm, display in arms.items():
        payload = read_json(DOWNSTREAM / "efficiency" / f"{arm}.json")
        require_completed(payload, "r2mt_matched_cuda_efficiency_v1")
        for batch_size in (1, 8):
            profile = payload["measurement"]["profiles"][f"batch_{batch_size}"]
            total = profile["latency_per_sample_ms"]
            stage = profile["stage_latency_per_sample_ms"]

            def stage_mean(name: str) -> float | str:
                return float(stage[name]["mean_ms"]) if name in stage else ""

            rows.append(
                {
                    "arm": arm,
                    "display_name": display,
                    "batch_size": batch_size,
                    "parameters": int(payload["parameters"]["total_parameters"]),
                    "registered_flops_per_sample": int(profile["flops"]["dispatcher_supported_per_sample"]),
                    "mean_latency_ms_per_sample": float(total["mean_ms"]),
                    "p95_latency_ms_per_sample": float(total["p95_ms"]),
                    "throughput_samples_per_second": float(profile["throughput_samples_per_second"]),
                    "peak_allocated_mib": float(profile["cuda_memory"]["peak_allocated_mib"]),
                    "host_to_device_ms": stage_mean("host_to_device"),
                    "raw_anchor_forward_ms": stage_mean("raw_anchor_forward"),
                    "sarn_anchor_forward_ms": stage_mean("sarn_anchor_forward"),
                    "risk_transport_ms": stage_mean("mett_correction"),
                    "posterior_validation_ms": stage_mean("posterior_moment_validation"),
                    "warmup_iterations": int(profile["warmup_iterations"]),
                    "timed_iterations": int(profile["timed_iterations"]),
                    "device": payload["environment"]["device_name"],
                    "precision": payload["measurement"]["autocast"],
                    "scope": "model-only; SARN materialization, detector, and OCR excluded",
                }
            )
    write_csv(
        HERE / "source_r2mt_efficiency.csv",
        rows,
        list(rows[0]),
    )
    return rows


def load_repeatability_rows() -> list[dict[str, Any]]:
    payload = read_json(DOWNSTREAM / "repeat" / "natural_repeat.json")
    require_completed(payload, "r2mt_natural_repeat_stability_v1")
    cohort = payload["cohort"]
    summary = payload["summary"]["methods"]["mett"]["three_seed"]
    paired = payload["summary"]["paired_comparisons"]["mett_minus_raw_anchor"]
    rows = [
        {
            "retained_samples": int(cohort["retained_samples"]),
            "distinct_source_images": int(cohort["retained_distinct_source_images"]),
            "exact_reading_units": int(cohort["retained_exact_reading_units"]),
            "physical_groups": int(cohort["retained_physical_groups"]),
            "full_denominator_nmae_mean": float(summary["full_denominator_nmae"]["mean"]),
            "full_denominator_nmae_sample_sd": float(summary["full_denominator_nmae"]["sample_sd"]),
            "mean_within_unit_population_sd": float(summary["mean_within_unit_prediction_sd_population"]["mean"]),
            "mean_within_unit_population_sd_sample_sd": float(summary["mean_within_unit_prediction_sd_population"]["sample_sd"]),
            "mean_within_unit_range": float(summary["mean_within_unit_prediction_range"]["mean"]),
            "mean_within_unit_range_sample_sd": float(summary["mean_within_unit_prediction_range"]["sample_sd"]),
            "r2mt_minus_raw_anchor_nmae": float(
                paired["physical_group_cluster_bootstrap_95ci"]["full_denominator_nmae"]["point_estimate"]
            ),
            "interpretation": "identity fallback on clean natural captures",
        }
    ]
    write_csv(
        HERE / "source_r2mt_repeatability.csv",
        rows,
        list(rows[0]),
    )
    return rows


def build_architecture() -> None:
    fig = plt.figure(figsize=(7.10, 6.25))
    grid = fig.add_gridspec(3, 1, left=0.035, right=0.985, bottom=0.035, top=0.965, hspace=0.30)

    ax_a = fig.add_subplot(grid[0])
    schematic_axis(ax_a, "R²MT-Net observations and shared anchor", "a")
    box(ax_a, (0.01, 0.39), (0.10, 0.25), "Raw ROI\n$I_r$", face="white", edge=COLORS["gray"], fontsize=7.3, weight="bold")
    box(ax_a, (0.15, 0.39), (0.13, 0.25), "SARN\n$M,H_{r\\to s}$", face=COLORS["teal_soft"], edge=COLORS["teal"], fontsize=7.3, weight="bold")
    box(ax_a, (0.32, 0.50), (0.19, 0.31), "Shared ResNet-18\nRaw pass\n$z_r,F_r^{8},F_r^{16}$", face=COLORS["gray_soft"], edge=COLORS["gray"], fontsize=7.3, weight="bold")
    box(ax_a, (0.32, 0.10), (0.19, 0.31), "Same ResNet-18\nSARN pass\n$z_s,F_s^{8},F_s^{16}$", face=COLORS["teal_soft"], edge=COLORS["teal"], fontsize=7.3, weight="bold")
    box(ax_a, (0.57, 0.49), (0.17, 0.25), "Moment-exact lift\n$q_r,\\mu_r$", face=COLORS["blue_soft"], edge=COLORS["blue"], fontsize=7.3, weight="bold")
    box(ax_a, (0.57, 0.13), (0.17, 0.25), "Moment-exact lift\n$q_s,\\mu_s$", face=COLORS["teal_soft"], edge=COLORS["teal"], fontsize=7.3, weight="bold")
    box(ax_a, (0.80, 0.31), (0.18, 0.32), "Inputs to base transport\nposteriors + multi-scale\nfeatures + support geometry", face=COLORS["orange_soft"], edge=COLORS["orange"], fontsize=6.2, weight="bold")
    arrow(ax_a, (0.11, 0.52), (0.32, 0.66))
    arrow(ax_a, (0.11, 0.47), (0.15, 0.51), color=COLORS["teal"])
    arrow(ax_a, (0.28, 0.50), (0.32, 0.26), color=COLORS["teal"])
    arrow(ax_a, (0.51, 0.66), (0.57, 0.62), color=COLORS["blue"])
    arrow(ax_a, (0.51, 0.26), (0.57, 0.25), color=COLORS["teal"])
    arrow(ax_a, (0.74, 0.61), (0.80, 0.54), color=COLORS["orange"])
    arrow(ax_a, (0.74, 0.25), (0.80, 0.40), color=COLORS["orange"])
    ax_a.text(0.415, 0.91, "one parameter set; two forward passes", ha="center", va="center", fontsize=5.6, color=COLORS["muted"])

    ax_b = fig.add_subplot(grid[1])
    schematic_axis(ax_b, "Base relation-encoded moment-exact transport", "b")
    box(ax_b, (0.01, 0.32), (0.13, 0.34), "Aligned features\n$F_r^l,\\,H^{-1}F_s^l$\n$l\\in\\{8,16\\}$", face=COLORS["gray_soft"], edge=COLORS["gray"], fontsize=7.3, weight="bold")
    box(ax_b, (0.18, 0.25), (0.20, 0.48), "Per-scale relation code\n$[u_r,u_s,u_s-u_r,$\n$|u_s-u_r|,u_r\\odot u_s,$\n$s,\\cos(u_r,u_s)]$", face=COLORS["orange_soft"], edge=COLORS["orange"], fontsize=7.3, weight="bold")
    box(ax_b, (0.42, 0.52), (0.16, 0.25), "Relation memory\nstride 8 + stride 16", face=COLORS["orange_soft"], edge=COLORS["orange"], weight="bold")
    box(ax_b, (0.42, 0.17), (0.16, 0.25), "Geometry token\n$8\\,\\Delta H + 2$ support\n$\\Delta\\mu,\\ln(v_s/v_r),\\mathrm{JS}$", face=COLORS["teal_soft"], edge=COLORS["teal"], fontsize=7.3, weight="bold")
    box(ax_b, (0.61, 0.30), (0.18, 0.38), "128-bin decoder\nprogress queries +\nrelation/geometry memory\n$\\Rightarrow\\delta_b$", face=COLORS["purple_soft"], edge=COLORS["purple"], fontsize=7.3, weight="bold")
    box(ax_b, (0.82, 0.30), (0.17, 0.38), "Single KL projection\n$q_b\\propto q_s e^{\\lambda x}$\nmatch $\\sum_k q_{b,k}x_k=\\mu_b$", face=COLORS["blue_soft"], edge=COLORS["blue"], fontsize=7.3, weight="bold")
    arrow(ax_b, (0.14, 0.49), (0.18, 0.49), color=COLORS["orange"])
    arrow(ax_b, (0.38, 0.55), (0.42, 0.64), color=COLORS["orange"])
    arrow(ax_b, (0.38, 0.43), (0.42, 0.29), color=COLORS["teal"])
    arrow(ax_b, (0.58, 0.64), (0.61, 0.56), color=COLORS["purple"])
    arrow(ax_b, (0.58, 0.29), (0.61, 0.42), color=COLORS["purple"])
    arrow(ax_b, (0.79, 0.49), (0.82, 0.49), color=COLORS["blue"])
    ax_b.text(0.50, 0.06, "If support, homography, or common aligned support is invalid: $q_b=q_r$ exactly.", ha="center", va="center", fontsize=7.3, color=COLORS["rose"], fontweight="bold")

    ax_c = fig.add_subplot(grid[2])
    schematic_axis(ax_c, "R²MT-Net representation-conditioned risk arbitration", "c")
    box(ax_c, (0.01, 0.32), (0.13, 0.36), "Base state\n$q_b,\\mu_b$\n$z_r,z_s,g$", face=COLORS["orange_soft"], edge=COLORS["orange"], fontsize=7.3, weight="bold")
    box(ax_c, (0.18, 0.62), (0.17, 0.23), "Mean-risk specialist\nδ1 in [−0.05, 0.05]", face=COLORS["blue_soft"], edge=COLORS["blue"], fontsize=6.3, weight="bold")
    box(ax_c, (0.18, 0.36), (0.17, 0.23), "Tail-risk specialist\nδ2 in [−0.05, 0.05]", face=COLORS["purple_soft"], edge=COLORS["purple"], fontsize=6.3, weight="bold")
    box(ax_c, (0.18, 0.10), (0.17, 0.23), "Combined-risk specialist\nδ3 in [−0.05, 0.05]", face=COLORS["rose_soft"], edge=COLORS["rose"], fontsize=6.1, weight="bold")
    box(ax_c, (0.41, 0.32), (0.19, 0.36), "Representation router\n[zr, zs, zs − zr, |zs − zr|,\nzr × zs, geometry, moments, δ]", face=COLORS["teal_soft"], edge=COLORS["teal"], fontsize=6.1, weight="bold")
    box(ax_c, (0.64, 0.38), (0.14, 0.25), "Conservative weights\nw = π + 0.5(w̃ − π)\nπ = (0.475, 0.280, 0.245)", face=COLORS["gray_soft"], edge=COLORS["gray"], fontsize=5.8, weight="bold")
    box(ax_c, (0.82, 0.31), (0.17, 0.39), "One target mean\nμ* = μb + Σ wi δi\nthen one exact transport\nq* = normalize[qb exp(λx)]", face=COLORS["blue_soft"], edge=COLORS["blue"], fontsize=5.8, weight="bold")
    for y in (0.735, 0.475, 0.215):
        arrow(ax_c, (0.14, 0.50), (0.18, y), color=COLORS["orange"], connectionstyle="arc3,rad=0.08")
        arrow(ax_c, (0.35, y), (0.41, 0.50), color=COLORS["teal"], connectionstyle="arc3,rad=-0.08")
    arrow(ax_c, (0.60, 0.50), (0.64, 0.50), color=COLORS["teal"])
    arrow(ax_c, (0.78, 0.50), (0.82, 0.50), color=COLORS["blue"])
    ax_c.text(0.50, 0.02, "R² denotes the two added principles: representation-conditioned routing and risk-specialized moment transport.", ha="center", va="bottom", fontsize=5.7, color=COLORS["muted"])
    save_all(fig, "r2mt_architecture")


def build_ocr_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.10, 2.35))
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.08, top=0.88)
    schematic_axis(ax, "OCR-enabled end-to-end image-to-physical-reading structure", "a")
    box(ax, (0.01, 0.38), (0.12, 0.27), "Field image", weight="bold")
    box(ax, (0.17, 0.38), (0.13, 0.27), "Meter\nlocalization", face=COLORS["gray_soft"], edge=COLORS["gray"], weight="bold")
    box(ax, (0.36, 0.60), (0.19, 0.25), "R²MT-Net\nnormalized progress $\\hat p$", face=COLORS["blue_soft"], edge=COLORS["blue"], weight="bold")
    box(ax, (0.36, 0.16), (0.19, 0.25), "OCR range branch\ntext regions + recognition\n$\\hat s_{min},\\hat s_{max}$", face=COLORS["teal_soft"], edge=COLORS["teal"], weight="bold")
    box(ax, (0.62, 0.38), (0.16, 0.27), "Validity checks\nordered endpoints\nunit consistency", face=COLORS["orange_soft"], edge=COLORS["orange"], weight="bold")
    box(ax, (0.84, 0.38), (0.15, 0.27), "Physical reading\n$\\hat y=\\hat s_{min}+\\hat p$\n$\\cdot(\\hat s_{max}-\\hat s_{min})$", face=COLORS["blue_soft"], edge=COLORS["blue"], weight="bold")
    arrow(ax, (0.13, 0.515), (0.17, 0.515))
    arrow(ax, (0.30, 0.54), (0.36, 0.72), color=COLORS["blue"])
    arrow(ax, (0.30, 0.49), (0.36, 0.29), color=COLORS["teal"])
    arrow(ax, (0.55, 0.72), (0.62, 0.57), color=COLORS["blue"])
    arrow(ax, (0.55, 0.29), (0.62, 0.46), color=COLORS["teal"])
    arrow(ax, (0.78, 0.515), (0.84, 0.515), color=COLORS["orange"])
    ax.text(0.50, 0.03, "Progress and OCR endpoints are combined only after endpoint ordering and unit-consistency checks.", ha="center", va="bottom", fontsize=5.8, color=COLORS["muted"])
    save_all(fig, "r2mt_ocr_pipeline")


def build_core_performance(rows: list[dict[str, Any]]) -> None:
    lookup = {(row["method"], row["metric"]): row for row in rows}
    methods = tuple(METHOD_COLORS)
    fig = plt.figure(figsize=(7.10, 5.05))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.45, 1.0],
        width_ratios=[1.20, 0.80],
        left=0.085,
        right=0.975,
        bottom=0.10,
        top=0.92,
        hspace=0.47,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(grid[0, :])
    x = np.arange(len(CONDITIONS))
    for method in methods:
        means = np.asarray([lookup[(method, condition)]["mean_nmae"] for condition in CONDITIONS]) * 100.0
        sds = np.asarray([lookup[(method, condition)]["sample_sd"] for condition in CONDITIONS]) * 100.0
        is_full = method == "R2MT-Net"
        ax_a.errorbar(
            x,
            means,
            yerr=sds,
            marker="D" if is_full else "o",
            markersize=5.1 if is_full else 3.6,
            linewidth=2.0 if is_full else 1.05,
            capsize=2.2,
            color=METHOD_COLORS[method],
            label=method.replace("R2MT", "R²MT"),
            markeredgecolor="white" if is_full else METHOD_COLORS[method],
            markeredgewidth=0.7 if is_full else 0.0,
            zorder=5 if is_full else 2,
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([CONDITION_LABELS[item] for item in CONDITIONS])
    ax_a.set_ylabel("NMAE (%FS)")
    ax_a.set_title(
        "R²MT-Net separates from matched endpoints as geometric severity increases",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    ax_a.text(
        0.0,
        1.015,
        "1,558 images · 14 held-out scene groups · mean ± sample SD across three independent fits",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=COLORS["muted"],
    )
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax_a.set_axisbelow(True)
    ax_a.legend(
        ncol=4,
        loc="upper left",
        frameon=False,
        handlelength=1.7,
        columnspacing=1.0,
    )
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[1, 0])
    comparators = methods[:-1]
    gains = np.asarray(
        [
            [
                100.0
                * (
                    lookup[(method, condition)]["mean_nmae"]
                    - lookup[("R2MT-Net", condition)]["mean_nmae"]
                )
                / lookup[(method, condition)]["mean_nmae"]
                for condition in CONDITIONS
            ]
            for method in comparators
        ],
        dtype=np.float64,
    )
    gain_cmap = LinearSegmentedColormap.from_list(
        "r2mt_gain", ("#F4F7F9", "#B9D7EC", "#135BA1")
    )
    image = ax_b.imshow(
        gains,
        cmap=gain_cmap,
        vmin=0.0,
        vmax=max(40.0, float(gains.max())),
        aspect="auto",
    )
    ax_b.set_xticks(np.arange(len(CONDITIONS)))
    ax_b.set_xticklabels([CONDITION_LABELS[item] for item in CONDITIONS], rotation=28, ha="right", rotation_mode="anchor")
    ax_b.set_yticks(np.arange(len(comparators)))
    ax_b.set_yticklabels([item.replace("MobileNetV3-Large", "MobileNetV3") for item in comparators])
    for row_index in range(gains.shape[0]):
        for column_index in range(gains.shape[1]):
            value = gains[row_index, column_index]
            ax_b.text(
                column_index,
                row_index,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if value > 27.0 else COLORS["ink"],
                fontweight="bold" if value > 25.0 else "normal",
            )
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_b, orientation="horizontal", fraction=0.08, pad=0.20, aspect=35)
    colorbar.set_label("Relative NMAE reduction by R²MT-Net")
    colorbar.outline.set_visible(False)
    ax_b.set_title("Condition-specific reduction versus each comparator", loc="left", fontweight="bold")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[1, 1])
    baseline = np.asarray([lookup[("EfficientNet-B0", condition)]["mean_nmae"] for condition in CONDITIONS]) * 100.0
    proposed = np.asarray([lookup[("R2MT-Net", condition)]["mean_nmae"] for condition in CONDITIONS]) * 100.0
    baseline_sd = np.asarray([lookup[("EfficientNet-B0", condition)]["sample_sd"] for condition in CONDITIONS]) * 100.0
    proposed_sd = np.asarray([lookup[("R2MT-Net", condition)]["sample_sd"] for condition in CONDITIONS]) * 100.0
    y = np.arange(len(CONDITIONS), dtype=np.float64)[::-1]
    for yi, efficient, r2mt in zip(y, baseline, proposed):
        ax_c.plot(
            [r2mt, efficient],
            [yi, yi],
            color=COLORS["grid"],
            linewidth=2.1,
            zorder=0,
        )
    ax_c.errorbar(
        baseline,
        y - 0.10,
        xerr=baseline_sd,
        fmt="o",
        markersize=3.8,
        color=COLORS["orange"],
        markeredgecolor="white",
        markeredgewidth=0.5,
        elinewidth=0.8,
        capsize=1.7,
        label="EfficientNet-B0",
        zorder=2,
    )
    ax_c.errorbar(
        proposed,
        y + 0.10,
        xerr=proposed_sd,
        fmt="D",
        markersize=4.5,
        color=COLORS["blue"],
        markeredgecolor="white",
        markeredgewidth=0.6,
        elinewidth=0.9,
        capsize=1.7,
        label="R²MT-Net",
        zorder=3,
    )
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([CONDITION_LABELS[item] for item in CONDITIONS])
    ax_c.set_xlabel("NMAE (%FS; lower is better)")
    ax_c.set_title("Matched-condition gaps", loc="left", fontweight="bold")
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    ax_c.set_axisbelow(True)
    ax_c.legend(loc="upper right", fontsize=5.1, handlelength=1.0)
    panel_label(ax_c, "c")
    save_all(fig, "r2mt_core_performance")


def build_ablation(rows: list[dict[str, Any]], effects: list[dict[str, Any]], routing: list[dict[str, Any]]) -> None:
    variants = (
        "Base relation transport",
        "Mean-risk specialist",
        "Tail-risk specialist",
        "Combined-risk specialist",
        "Fixed three-risk prior",
        "Adaptive multi-risk routing",
        "w/o deep representations",
        "w/o risk-imitation loss",
        "posterior probability mixture",
    )
    display_aliases = {
        "ReMST base": "Base relation transport",
        "Mean-risk RCMT": "Mean-risk specialist",
    }
    lookup = {
        (display_aliases.get(row["method"], row["method"]), row["metric"]): row
        for row in rows
    }
    fig = plt.figure(figsize=(7.10, 5.75))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.28, 1.0],
        width_ratios=[1.48, 0.72],
        left=0.145,
        right=0.975,
        bottom=0.10,
        top=0.92,
        hspace=0.46,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(grid[0, :])
    full = np.asarray(
        [lookup[("R2MT-Net", condition)]["mean_nmae"] for condition in CONDITIONS],
        dtype=np.float64,
    )
    penalties = np.asarray(
        [
            [100.0 * (lookup[(variant, condition)]["mean_nmae"] - full[index]) for index, condition in enumerate(CONDITIONS)]
            for variant in variants
        ],
        dtype=np.float64,
    )
    bound = max(abs(float(penalties.min())), abs(float(penalties.max())))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    heat = ax_a.imshow(penalties, cmap="RdBu_r", norm=norm, aspect="auto")
    ax_a.set_xticks(np.arange(len(CONDITIONS)))
    ax_a.set_xticklabels([CONDITION_LABELS[item] for item in CONDITIONS])
    ax_a.set_yticks(np.arange(len(variants)))
    ax_a.set_yticklabels([item.replace("R2MT", "R²MT").replace("posterior probability mixture", "posterior mixture") for item in variants])
    for row_index in range(penalties.shape[0]):
        for column_index in range(penalties.shape[1]):
            value = penalties[row_index, column_index]
            rgba = plt.get_cmap("RdBu_r")(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax_a.text(
                column_index,
                row_index,
                "≈0" if abs(value) < 0.0005 else f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if luminance < 0.48 else COLORS["ink"],
                fontweight="bold" if abs(value) >= 0.05 else "normal",
            )
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(heat, ax=ax_a, orientation="vertical", fraction=0.025, pad=0.018)
    colorbar.set_label("NMAE penalty versus full R²MT-Net (pp)")
    colorbar.outline.set_visible(False)
    ax_a.set_title("Every ablation is resolved across all six image conditions", loc="left", fontweight="bold", pad=8)
    ax_a.text(0.0, 1.015, "Positive values are worse than the complete model; cells are mean differences across three fits", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=6.0, color=COLORS["muted"])
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[1, 0])
    paired_names = (
        "Base relation transport",
        "Mean-risk RCMT",
        "posterior probability mixture",
    )
    effect_lookup = {
        (("Base relation transport" if row["comparator"] == "ReMST base" else row["comparator"]), row["metric"]): row
        for row in effects
    }
    paired_specs = (
        ("Base relation transport", COLORS["teal"], "o", "internal relation foundation"),
        ("Mean-risk RCMT", COLORS["blue"], "D", "best single-risk head"),
        ("posterior probability mixture", COLORS["rose"], "s", "posterior mixture"),
    )
    xb = np.arange(len(CONDITIONS), dtype=np.float64)
    offsets = (-0.12, 0.0, 0.12)
    for offset, (name, color, marker, label) in zip(offsets, paired_specs):
        estimates: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for metric in CONDITIONS:
            row = effect_lookup[(name, metric)]
            estimate = float(row["reduction_pp"])
            estimates.append(estimate)
            lows.append(float(row["ci_low_pp"]))
            highs.append(float(row["ci_high_pp"]))
        values = np.asarray(estimates)
        ax_b.errorbar(
            xb + offset,
            values,
            yerr=[values - np.asarray(lows), np.asarray(highs) - values],
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=4.3,
            linewidth=1.0,
            capsize=2.0,
            label=label,
        )
    ax_b.axhline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([CONDITION_LABELS[item] for item in CONDITIONS], rotation=20, ha="right", rotation_mode="anchor")
    ax_b.set_ylabel("NMAE reduction by R²MT-Net (pp)")
    ax_b.set_title("Paired scene-cluster effects by condition", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax_b.set_axisbelow(True)
    ax_b.legend(ncol=3, loc="upper left", frameon=False, handletextpad=0.4, columnspacing=0.8)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[1, 1])
    route_lookup = {(row["route"], row["metric"]): row for row in routing}
    route_order = ("fixed", "conservative", "adaptive")
    conservative = np.asarray([route_lookup[("conservative", condition)]["mean_nmae"] for condition in CONDITIONS])
    route_delta = np.asarray(
        [
            [100_000.0 * (route_lookup[(route, condition)]["mean_nmae"] - conservative[index]) for index, condition in enumerate(CONDITIONS)]
            for route in route_order
        ]
    )
    route_bound = max(1.0, abs(float(route_delta.min())), abs(float(route_delta.max())))
    route_heat = ax_c.imshow(route_delta, cmap="PiYG_r", norm=TwoSlopeNorm(vmin=-route_bound, vcenter=0.0, vmax=route_bound), aspect="auto")
    ax_c.set_xticks(np.arange(len(CONDITIONS)))
    ax_c.set_xticklabels([CONDITION_LABELS[item].replace("Persp.-", "P") for item in CONDITIONS], rotation=45, ha="right", rotation_mode="anchor")
    ax_c.set_yticks(np.arange(3))
    ax_c.set_yticklabels(["Fixed α=0", "Conservative α=0.5", "Adaptive α=1"])
    for row_index in range(route_delta.shape[0]):
        for column_index in range(route_delta.shape[1]):
            ax_c.text(column_index, row_index, f"{route_delta[row_index, column_index]:+.2f}", ha="center", va="center", fontsize=5.2, color=COLORS["ink"])
    ax_c.tick_params(length=0)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    colorbar_route = fig.colorbar(route_heat, ax=ax_c, orientation="horizontal", fraction=0.08, pad=0.27, aspect=25)
    colorbar_route.set_label("Δ NMAE vs α=0.5 (10^-3 pp)")
    colorbar_route.outline.set_visible(False)
    ax_c.set_title("Routing strength is a small refinement", loc="left", fontweight="bold")
    panel_label(ax_c, "c")
    save_all(fig, "r2mt_ablation")


def build_industrial_validation(field_rows: list[dict[str, Any]]) -> None:
    """Show the matched four-model industrial real-photo comparison."""
    field_lookup = {
        (str(row["metric"]), str(row["method"])): row for row in field_rows
    }
    fig = plt.figure(figsize=(7.10, 6.55))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.38, 0.72, 0.68],
        left=0.12,
        right=0.97,
        bottom=0.075,
        top=0.94,
        hspace=0.72,
    )

    ax_a = fig.add_subplot(grid[0])
    x = np.arange(len(CONDITIONS), dtype=np.float64)
    ax_a.axvspan(2.5, 5.5, color=COLORS["teal_soft"], alpha=0.58, zorder=0)
    method_specs = (
        ("Direct-ResNet18", "o", 1.05, 3.8),
        ("EfficientNet-B0", "s", 1.05, 3.8),
        ("MobileNetV3-Large", "^", 1.05, 4.0),
        ("R²MT-Net", "D", 2.0, 5.2),
    )
    for method, marker, linewidth, markersize in method_specs:
        means = np.asarray(
            [
                100.0 * float(field_lookup[(metric, method)]["mean_nmae"])
                for metric in CONDITIONS
            ]
        )
        sds = np.asarray(
            [
                100.0 * float(field_lookup[(metric, method)]["sample_sd"])
                for metric in CONDITIONS
            ]
        )
        color = METHOD_COLORS["R2MT-Net" if method == "R²MT-Net" else method]
        ax_a.errorbar(
            x,
            means,
            yerr=sds,
            color=color,
            marker=marker,
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=linewidth,
            capsize=2.0,
            elinewidth=0.9,
            alpha=1.0 if method == "R²MT-Net" else 0.88,
            label=method,
            zorder=4 if method == "R²MT-Net" else 2,
        )
    ax_a.axvline(2.5, color=COLORS["teal"], linewidth=0.7, linestyle="--")
    ax_a.text(
        2.62,
        0.97,
        "projective-risk conditions",
        transform=ax_a.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=5.5,
        color=COLORS["teal"],
        fontweight="bold",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [CONDITION_LABELS[item] for item in CONDITIONS],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax_a.set_ylabel("Industrial NMAE (%FS; lower is better)")
    ax_a.set_title(
        "Industrial real photos reveal condition-dependent transfer across four models",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    ax_a.text(
        0.0,
        1.015,
        "1,395 labeled ROIs · 52 source groups · identical condition pixels · mean ± sample SD across three fits",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=COLORS["muted"],
    )
    ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax_a.set_axisbelow(True)
    ax_a.legend(
        ncol=4,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.20),
        columnspacing=1.25,
        handlelength=1.5,
    )
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[1])
    comparators = ("Direct-ResNet18", "EfficientNet-B0", "MobileNetV3-Large")
    effects = np.asarray(
        [
            [
                float(field_lookup[(metric, comparator)]["r2mt_paired_reduction_pp"])
                for metric in CONDITIONS
            ]
            for comparator in comparators
        ]
    )
    significance = np.asarray(
        [
            [
                bool(field_lookup[(metric, comparator)]["ci95_excludes_zero"])
                for metric in CONDITIONS
            ]
            for comparator in comparators
        ]
    )
    effect_limit = float(np.ceil(np.max(np.abs(effects))))
    effect_cmap = LinearSegmentedColormap.from_list(
        "industrial_effect",
        [COLORS["orange"], "#FFFFFF", COLORS["blue"]],
    )
    image_b = ax_b.imshow(
        effects,
        aspect="auto",
        cmap=effect_cmap,
        norm=TwoSlopeNorm(vmin=-effect_limit, vcenter=0.0, vmax=effect_limit),
    )
    for row_index in range(effects.shape[0]):
        for column_index in range(effects.shape[1]):
            value = float(effects[row_index, column_index])
            ax_b.text(
                column_index,
                row_index,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=5.7,
                fontweight="bold" if significance[row_index, column_index] else "normal",
                color="white" if abs(value) > 0.58 * effect_limit else COLORS["ink"],
            )
            if significance[row_index, column_index]:
                ax_b.add_patch(
                    Rectangle(
                        (column_index - 0.47, row_index - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        edgecolor=COLORS["ink"],
                        linewidth=0.75,
                    )
                )
    ax_b.set_xticks(np.arange(len(CONDITIONS)))
    ax_b.set_xticklabels(
        [CONDITION_LABELS[item] for item in CONDITIONS],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    ax_b.set_yticks(np.arange(len(comparators)))
    ax_b.set_yticklabels(comparators)
    ax_b.set_title(
        "Paired group effects vary by condition and comparator",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    ax_b.text(
        0.0,
        1.02,
        "bold outline: 95% group-bootstrap CI excludes zero",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.2,
        color=COLORS["muted"],
    )
    colorbar_b = fig.colorbar(image_b, ax=ax_b, fraction=0.052, pad=0.035)
    colorbar_b.set_label("R²MT-Net NMAE reduction (percentage points)", fontsize=5.6)
    colorbar_b.ax.tick_params(labelsize=5.2)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[2])
    y = np.arange(len(comparators), dtype=np.float64)[::-1]
    aggregate_specs = (
        ("all_conditions", "All six", -0.12, COLORS["gray"], "o"),
        ("projective_pooled", "Projective pool", 0.12, COLORS["teal"], "D"),
    )
    all_intervals: list[float] = []
    for metric, label, offset, color, marker in aggregate_specs:
        estimates = np.asarray(
            [
                float(field_lookup[(metric, comparator)]["r2mt_paired_reduction_pp"])
                for comparator in comparators
            ]
        )
        lows = np.asarray(
            [
                float(field_lookup[(metric, comparator)]["ci95_low_pp"])
                for comparator in comparators
            ]
        )
        highs = np.asarray(
            [
                float(field_lookup[(metric, comparator)]["ci95_high_pp"])
                for comparator in comparators
            ]
        )
        all_intervals.extend(lows.tolist())
        all_intervals.extend(highs.tolist())
        ax_c.errorbar(
            estimates,
            y + offset,
            xerr=[estimates - lows, highs - estimates],
            fmt=marker,
            markersize=4.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            elinewidth=1.0,
            capsize=2.0,
            label=label,
            zorder=3,
        )
    ax_c.axvline(0.0, color=COLORS["ink"], linewidth=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(comparators)
    interval_limit = max(abs(min(all_intervals)), abs(max(all_intervals))) + 0.7
    ax_c.set_xlim(-interval_limit, interval_limit)
    ax_c.set_xlabel("Paired NMAE reduction (percentage points)")
    ax_c.set_title(
        "Aggregate effects retain comparator identity",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    ax_c.text(
        0.0,
        1.02,
        "positive favors R²MT-Net · mean row error across fits · 95% group-bootstrap CI",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.2,
        color=COLORS["muted"],
    )
    ax_c.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=False, loc="lower right", handlelength=1.3)
    panel_label(ax_c, "c")
    save_all(fig, "r2mt_industrial_validation")


def build_vdn_validation(vdn_rows: list[dict[str, Any]]) -> None:
    """Separate the annotation-assisted VDN reference from deployable models."""
    lookup = {str(row["metric"]): row for row in vdn_rows}
    fig = plt.figure(figsize=(7.10, 3.05))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.16, 1.0],
        left=0.105,
        right=0.985,
        bottom=0.18,
        top=0.86,
        wspace=0.43,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    y = np.arange(len(CONDITIONS), dtype=np.float64)[::-1]
    for yi, metric in zip(y, CONDITIONS):
        values = [
            100.0 * float(lookup[metric]["r2mt_mean_nmae"]),
            100.0 * float(lookup[metric]["efficientnet_mean_nmae"]),
            100.0 * float(lookup[metric]["vdn_annotation_nmae"]),
        ]
        ax_a.plot(
            [min(values), max(values)],
            [yi, yi],
            color=COLORS["grid"],
            linewidth=2.0,
            zorder=0,
        )
    for label, mean_key, sd_key, color, marker, offset in (
        (
            "R²MT-Net",
            "r2mt_mean_nmae",
            "r2mt_sample_sd",
            COLORS["blue"],
            "D",
            0.11,
        ),
        (
            "SARN + EfficientNet-B0",
            "efficientnet_mean_nmae",
            "efficientnet_sample_sd",
            COLORS["orange"],
            "o",
            -0.11,
        ),
    ):
        means = np.asarray(
            [100.0 * float(lookup[metric][mean_key]) for metric in CONDITIONS]
        )
        sds = np.asarray(
            [100.0 * float(lookup[metric][sd_key]) for metric in CONDITIONS]
        )
        ax_a.errorbar(
            means,
            y + offset,
            xerr=sds,
            fmt=marker,
            markersize=4.7,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            elinewidth=0.9,
            capsize=2,
            label=label,
            zorder=3,
        )
    vdn_values = np.asarray(
        [100.0 * float(lookup[metric]["vdn_annotation_nmae"]) for metric in CONDITIONS]
    )
    ax_a.scatter(
        vdn_values,
        y,
        marker="^",
        s=31,
        color=COLORS["gray"],
        edgecolor="white",
        linewidth=0.6,
        label="VDN annotation reference",
        zorder=3,
    )
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([CONDITION_LABELS[item] for item in CONDITIONS])
    ax_a.set_xlabel("NMAE (%FS; lower is better)")
    ax_a.set_title(
        "Same-pixel external-method comparison",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    ax_a.text(
        0.0,
        1.02,
        "129 samples · six conditions · 774 identical condition rows",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        color=COLORS["muted"],
    )
    ax_a.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.legend(
        frameon=False,
        loc="upper right",
        ncol=1,
        fontsize=5.2,
        handlelength=1.1,
    )
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[0, 1])
    comparator_specs = (
        (
            "SARN + EfficientNet-B0",
            "r2mt_reduction_vs_efficientnet_pp",
            "r2mt_reduction_vs_efficientnet_ci_low_pp",
            "r2mt_reduction_vs_efficientnet_ci_high_pp",
        ),
        (
            "VDN annotation reference",
            "r2mt_reduction_vs_vdn_pp",
            "r2mt_reduction_vs_vdn_ci_low_pp",
            "r2mt_reduction_vs_vdn_ci_high_pp",
        ),
    )
    effects = np.asarray(
        [
            [float(lookup[metric][value_key]) for metric in CONDITIONS]
            for _, value_key, _, _ in comparator_specs
        ]
    )
    significant = np.asarray(
        [
            [
                float(lookup[metric][low_key]) > 0.0
                or float(lookup[metric][high_key]) < 0.0
                for metric in CONDITIONS
            ]
            for _, _, low_key, high_key in comparator_specs
        ]
    )
    effect_limit = float(np.ceil(np.max(np.abs(effects))))
    cmap = LinearSegmentedColormap.from_list(
        "vdn_effect",
        [COLORS["orange"], "#FFFFFF", COLORS["blue"]],
    )
    image_b = ax_b.imshow(
        effects,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-effect_limit, vcenter=0.0, vmax=effect_limit),
    )
    for row_index in range(effects.shape[0]):
        for column_index in range(effects.shape[1]):
            value = float(effects[row_index, column_index])
            ax_b.text(
                column_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold" if significant[row_index, column_index] else "normal",
                color="white" if abs(value) > 0.60 * effect_limit else COLORS["ink"],
            )
            if significant[row_index, column_index]:
                ax_b.add_patch(
                    Rectangle(
                        (column_index - 0.47, row_index - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        edgecolor=COLORS["ink"],
                        linewidth=0.75,
                    )
                )
    ax_b.set_xticks(np.arange(len(CONDITIONS)))
    ax_b.set_xticklabels(
        [CONDITION_LABELS[item] for item in CONDITIONS],
        rotation=38,
        ha="right",
        rotation_mode="anchor",
    )
    ax_b.set_yticks(np.arange(len(comparator_specs)))
    ax_b.set_yticklabels([item[0] for item in comparator_specs])
    ax_b.set_title(
        "Condition-wise paired reductions",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    ax_b.text(
        0.0,
        1.02,
        "positive favors R²MT-Net · outline: 95% scene-bootstrap CI excludes zero",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.2,
        color=COLORS["muted"],
    )
    colorbar_b = fig.colorbar(image_b, ax=ax_b, fraction=0.052, pad=0.035)
    colorbar_b.set_label("NMAE reduction (percentage points)", fontsize=5.6)
    colorbar_b.ax.tick_params(labelsize=5.2)
    ax_b.text(
        0.0,
        -0.25,
        "VDN is a single annotation-assisted checkpoint and is not deployable from an image alone.",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color=COLORS["rose"],
    )
    panel_label(ax_b, "b")
    save_all(fig, "r2mt_vdn_validation")


def build_efficiency_profile(efficiency_rows: list[dict[str, Any]]) -> None:
    """Expose latency composition and the compute/parameter trade-off."""

    lookup = {
        (str(row["arm"]), int(row["batch_size"])): row
        for row in efficiency_rows
    }
    fig = plt.figure(figsize=(7.10, 4.70))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.18, 1.0],
        width_ratios=[1.15, 0.85],
        left=0.095,
        right=0.975,
        bottom=0.11,
        top=0.90,
        hspace=0.72,
        wspace=0.38,
    )

    ax_a = fig.add_subplot(grid[0, :])
    profiles = [lookup[("full_mett", 1)], lookup[("full_mett", 8)]]
    y = np.asarray([1.0, 0.0])
    components = (
        ("Host→device", "host_to_device_ms", COLORS["gray"]),
        ("Raw anchor", "raw_anchor_forward_ms", COLORS["blue"]),
        ("SARN anchor", "sarn_anchor_forward_ms", COLORS["teal"]),
        ("Relation + risk transport", "risk_transport_ms", COLORS["purple"]),
        ("Posterior validation", "posterior_validation_ms", COLORS["orange"]),
    )
    left = np.zeros(2, dtype=np.float64)
    for label, key, color in components:
        values = np.asarray([float(row[key]) for row in profiles])
        ax_a.barh(y, values, left=left, height=0.48, color=color, label=label)
        left += values
    totals = np.asarray([float(row["mean_latency_ms_per_sample"]) for row in profiles])
    overhead = np.maximum(totals - left, 0.0)
    ax_a.barh(y, overhead, left=left, height=0.48, color=COLORS["gray_soft"], edgecolor=COLORS["gray"], linewidth=0.4, label="Other synchronization")
    for yi, row in zip(y, profiles):
        total = float(row["mean_latency_ms_per_sample"])
        throughput = float(row["throughput_samples_per_second"])
        ax_a.text(total + 0.32, yi, f"{total:.2f} ms · {throughput:.1f} images/s", ha="left", va="center", fontsize=6.0, fontweight="bold", color=COLORS["ink"])
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(["Batch 1", "Batch 8"])
    ax_a.set_xlim(0.0, 29.0)
    ax_a.set_xlabel("Mean model-only latency per sample (ms)")
    ax_a.set_title("Risk transport, not parameter count, dominates batch-1 cost", loc="left", fontweight="bold", pad=14)
    ax_a.text(0.0, 1.02, "RTX 4060 · BF16 · 256×256 inputs · 20 warm-ups + 100 timed iterations", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=6.0, color=COLORS["muted"])
    ax_a.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax_a.set_axisbelow(True)
    ax_a.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.44), frameon=False, columnspacing=0.9, handlelength=1.2)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(grid[1, 0])
    arms = ("raw_only", "twin_endpoint", "full_mett")
    display = {"raw_only": "Raw anchor", "twin_endpoint": "Dual-view endpoint", "full_mett": "R²MT-Net"}
    arm_colors = {"raw_only": COLORS["gray"], "twin_endpoint": COLORS["teal"], "full_mett": COLORS["blue"]}
    yb = np.arange(len(arms), dtype=np.float64)[::-1]
    for yi, arm in zip(yb, arms):
        batch_1 = float(lookup[(arm, 1)]["mean_latency_ms_per_sample"])
        batch_8 = float(lookup[(arm, 8)]["mean_latency_ms_per_sample"])
        ax_b.plot([batch_8, batch_1], [yi, yi], color=arm_colors[arm], linewidth=2.0, alpha=0.75)
        ax_b.scatter(batch_1, yi, marker="o", s=27, color=arm_colors[arm], edgecolor="white", linewidth=0.6, zorder=3)
        ax_b.scatter(batch_8, yi, marker="D", s=27, color=arm_colors[arm], edgecolor="white", linewidth=0.6, zorder=3)
        ax_b.annotate(
            f"{batch_1:.2f}",
            (batch_1, yi),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=5.5,
            color=COLORS["ink"],
        )
        ax_b.annotate(
            f"{batch_8:.2f}",
            (batch_8, yi),
            xytext=(-5, 6),
            textcoords="offset points",
            ha="right",
            fontsize=5.5,
            color=COLORS["ink"],
        )
    latency_values = np.asarray(
        [
            float(lookup[(arm, batch)]["mean_latency_ms_per_sample"])
            for arm in arms
            for batch in (1, 8)
        ],
        dtype=np.float64,
    )
    if np.any(latency_values <= 0.0):
        raise ValueError("latencies must be strictly positive for log scaling")
    ax_b.set_xscale("log")
    ax_b.set_xticks([1.0, 10.0], labels=["1", "10"])
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([display[arm] for arm in arms])
    ax_b.set_ylim(-0.35, 2.35)
    ax_b.set_xlabel("Latency per sample (ms; log scale)")
    ax_b.set_title("Batching benefit across execution arms", loc="left", fontweight="bold")
    ax_b.text(
        0.98,
        0.98,
        "circle: Batch 1    diamond: Batch 8",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color=COLORS["muted"],
    )
    ax_b.grid(axis="x", color=COLORS["grid"], linewidth=0.5, which="both")
    ax_b.set_axisbelow(True)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[1, 1])
    for arm in arms:
        row = lookup[(arm, 1)]
        flops = float(row["registered_flops_per_sample"]) / 1.0e9
        parameters = float(row["parameters"]) / 1.0e6
        ax_c.scatter(flops, parameters, s=45 if arm == "full_mett" else 34, color=arm_colors[arm], edgecolor="white", linewidth=0.7, zorder=3)
        offsets = {
            "raw_only": (4, 6, "left"),
            "twin_endpoint": (-4, 6, "right"),
            "full_mett": (-4, 6, "right"),
        }
        dx, dy, alignment = offsets[arm]
        ax_c.annotate(
            display[arm],
            (flops, parameters),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            fontsize=5.6,
            color=COLORS["ink"],
        )
    ax_c.set_xlabel("Registered GFLOPs per sample")
    ax_c.set_ylabel("Unique parameters (M)")
    ax_c.set_title("Compute–parameter plane", loc="left", fontweight="bold")
    ax_c.grid(color=COLORS["grid"], linewidth=0.5)
    ax_c.set_axisbelow(True)
    ax_c.set_xlim(4.45, 9.85)
    ax_c.set_ylim(11.20, 11.75)
    panel_label(ax_c, "c")
    save_all(fig, "r2mt_efficiency")


def main() -> None:
    configure_style()
    main_rows = load_main_rows()
    ablation_rows, effects = load_ablation_rows()
    routing = load_routing_rows()
    field_rows = load_field_transfer_rows()
    vdn_rows = load_vdn_rows()
    load_ocr_accepted_rows()
    efficiency_rows = load_efficiency_rows()
    load_repeatability_rows()
    build_architecture()
    build_ocr_pipeline()
    build_core_performance(main_rows)
    build_ablation(ablation_rows, effects, routing)
    build_industrial_validation(field_rows)
    build_vdn_validation(vdn_rows)
    build_efficiency_profile(efficiency_rows)


if __name__ == "__main__":
    main()
