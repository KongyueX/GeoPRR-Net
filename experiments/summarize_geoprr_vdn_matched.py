"""Summarize the three-seed matched GeoPRR-versus-VDN comparison."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.evaluate_geoprr_vdn_matched import PROTOCOL as VDN_EVALUATION_PROTOCOL


PROTOCOL = "geoprr_vdn_matched_three_seed_summary_v1"
CONDITIONS = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
EXPECTED_SEEDS = (20262020, 20262021, 20262022)
FAILURE_ERROR = 1.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).resolve().open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(
                isinstance(value, dict),
                f"{path}:{line_number} is not an object",
            )
            rows.append(value)
    return rows


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("sample_id") or ""), str(row.get("condition") or "")


def _index_rows(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _key(row)
        _require(bool(key[0]) and key[1] in CONDITIONS, f"{label}: invalid row key")
        _require(key not in result, f"{label}: duplicate row {key}")
        result[key] = row
    _require(len(result) == 9_348, f"{label}: expected 9348 rows, got {len(result)}")
    _require(
        len({sample_id for sample_id, _ in result}) == 1_558,
        f"{label}: expected 1558 samples",
    )
    return result


def _metrics(errors: np.ndarray, passed: np.ndarray) -> dict[str, Any]:
    _require(errors.ndim == 1 and len(errors) > 0, "metric error vector is empty")
    _require(passed.shape == errors.shape, "metric pass vector differs")
    return {
        "rows": int(errors.size),
        "nmae": float(np.mean(errors)),
        "nmae_percent_fs": float(np.mean(errors) * 100.0),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "rmse_percent_fs": float(np.sqrt(np.mean(np.square(errors))) * 100.0),
        "p95_absolute_error": float(np.quantile(errors, 0.95)),
        "p95_percent_fs": float(np.quantile(errors, 0.95) * 100.0),
        "acc_at_1_percent_fs": float(np.mean(errors <= 0.01)),
        "acc_at_2_percent_fs": float(np.mean(errors <= 0.02)),
        "acc_at_5_percent_fs": float(np.mean(errors <= 0.05)),
        "coverage": float(np.mean(passed)),
        "failures": int(np.sum(~passed)),
    }


def _mean_sample_sd(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)),
    }


def _across_seed_metrics(
    per_seed: Mapping[int, Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    keys = (
        "nmae",
        "nmae_percent_fs",
        "rmse",
        "rmse_percent_fs",
        "p95_absolute_error",
        "p95_percent_fs",
        "acc_at_1_percent_fs",
        "acc_at_2_percent_fs",
        "acc_at_5_percent_fs",
        "coverage",
    )
    return {
        key: _mean_sample_sd([float(per_seed[seed][key]) for seed in EXPECTED_SEEDS])
        for key in keys
    }


def _scene_bootstrap_delta(
    delta: np.ndarray,
    scenes: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    selected_scenes = sorted(set(str(value) for value in scenes[mask]))
    _require(len(selected_scenes) == 14, "bootstrap scope does not contain 14 scenes")
    scene_sums = np.asarray(
        [float(np.sum(delta[mask & (scenes == scene)])) for scene in selected_scenes],
        dtype=np.float64,
    )
    scene_counts = np.asarray(
        [int(np.sum(mask & (scenes == scene))) for scene in selected_scenes],
        dtype=np.float64,
    )
    _require(np.all(scene_counts > 0), "bootstrap contains an empty scene")
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    offset = 0
    while offset < replicates:
        count = min(2_000, replicates - offset)
        draws = rng.integers(0, len(selected_scenes), size=(count, len(selected_scenes)))
        values[offset : offset + count] = (
            np.sum(scene_sums[draws], axis=1) / np.sum(scene_counts[draws], axis=1)
        )
        offset += count
    return {
        "delta_definition": "GeoPRR NMAE minus VDN NMAE",
        "observed": float(np.mean(delta[mask])),
        "observed_percent_fs": float(np.mean(delta[mask]) * 100.0),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "ci95_lower_percent_fs": float(np.quantile(values, 0.025) * 100.0),
        "ci95_upper_percent_fs": float(np.quantile(values, 0.975) * 100.0),
        "replicates": int(replicates),
        "scene_clusters": len(selected_scenes),
    }


def summarize(
    *,
    vdn_paths: Sequence[Path],
    geoprr_paths: Sequence[Path],
    pixel_reference_path: Path,
    output_path: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _require(len(vdn_paths) == 3, "exactly three VDN predictions are required")
    _require(len(geoprr_paths) == 3, "exactly three GeoPRR evaluations are required")
    _require(bootstrap_replicates >= 1_000, "at least 1000 bootstrap replicates required")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"summary output already exists: {output}")

    reference_payload = _read_json(pixel_reference_path)
    reference = _index_rows(
        reference_payload.get("per_sample_condition") or [],
        label="pixel reference",
    )
    ordered_keys = sorted(reference)
    scenes = np.asarray(
        [str(reference[key].get("scene_stem") or "") for key in ordered_keys],
        dtype=object,
    )
    conditions = np.asarray([key[1] for key in ordered_keys], dtype=object)
    targets = np.asarray(
        [float(reference[key]["normalized_target"]) for key in ordered_keys],
        dtype=np.float64,
    )
    _require(np.isfinite(targets).all(), "reference targets are non-finite")
    _require(len(set(scenes)) == 14 and all(scenes), "reference scene roster differs")

    vdn_errors: dict[int, np.ndarray] = {}
    vdn_passed: dict[int, np.ndarray] = {}
    vdn_sources: dict[int, str] = {}
    for path in vdn_paths:
        indexed = _index_rows(_read_jsonl(path), label=f"VDN {path}")
        _require(set(indexed) == set(reference), f"VDN key roster differs: {path}")
        methods = {str(row.get("method") or "") for row in indexed.values()}
        _require(len(methods) == 1, f"VDN method identity differs: {path}")
        method = methods.pop()
        try:
            seed = int(method.rsplit("_", 1)[-1])
        except ValueError as exc:
            raise ValueError(f"cannot read VDN seed from {method}") from exc
        _require(seed in EXPECTED_SEEDS and seed not in vdn_errors, "VDN seed roster differs")
        errors = np.empty(len(ordered_keys), dtype=np.float64)
        passed = np.empty(len(ordered_keys), dtype=bool)
        for index, key in enumerate(ordered_keys):
            row = indexed[key]
            _require(
                row.get("protocol") == VDN_EVALUATION_PROTOCOL,
                f"VDN protocol differs: {path}",
            )
            _require(
                row.get("condition_pixel_sha256")
                == reference[key].get("condition_pixel_sha256"),
                f"VDN pixel differs at {key}",
            )
            is_pass = row.get("status") == "pass"
            prediction = row.get("normalized_progress")
            if is_pass:
                _require(prediction is not None, f"VDN pass lacks prediction at {key}")
                value = float(prediction)
                _require(math.isfinite(value), f"VDN prediction is non-finite at {key}")
                errors[index] = abs(value - targets[index])
            else:
                errors[index] = FAILURE_ERROR
            passed[index] = is_pass
        vdn_errors[seed] = errors
        vdn_passed[seed] = passed
        vdn_sources[seed] = str(Path(path).resolve())
    _require(tuple(sorted(vdn_errors)) == EXPECTED_SEEDS, "VDN seeds are incomplete")

    geoprr_errors: dict[int, np.ndarray] = {}
    geoprr_passed: dict[int, np.ndarray] = {}
    geoprr_sources: dict[int, str] = {}
    for path in geoprr_paths:
        payload = _read_json(path)
        model = payload.get("model")
        _require(isinstance(model, Mapping), f"GeoPRR model metadata missing: {path}")
        seed = int(model.get("seed", -1))
        _require(
            seed in EXPECTED_SEEDS and seed not in geoprr_errors,
            "GeoPRR seed roster differs",
        )
        indexed = _index_rows(
            payload.get("per_sample_condition") or [],
            label=f"GeoPRR {path}",
        )
        _require(set(indexed) == set(reference), f"GeoPRR key roster differs: {path}")
        errors = np.empty(len(ordered_keys), dtype=np.float64)
        passed = np.ones(len(ordered_keys), dtype=bool)
        for index, key in enumerate(ordered_keys):
            row = indexed[key]
            _require(
                str(row.get("scene_stem")) == str(reference[key].get("scene_stem"))
                and math.isclose(
                    float(row.get("normalized_target")),
                    targets[index],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"GeoPRR reference differs at {key}",
            )
            metric = row.get("mett")
            _require(isinstance(metric, Mapping), f"GeoPRR metric missing at {key}")
            value = float(metric.get("absolute_error"))
            _require(math.isfinite(value) and value >= 0.0, f"GeoPRR error invalid at {key}")
            errors[index] = value
        geoprr_errors[seed] = errors
        geoprr_passed[seed] = passed
        geoprr_sources[seed] = str(Path(path).resolve())
    _require(tuple(sorted(geoprr_errors)) == EXPECTED_SEEDS, "GeoPRR seeds are incomplete")

    scopes: dict[str, np.ndarray] = {
        "all_conditions": np.ones(len(ordered_keys), dtype=bool),
        **{condition: conditions == condition for condition in CONDITIONS},
    }
    scope_reports: dict[str, Any] = {}
    for scope_index, (scope, mask) in enumerate(scopes.items()):
        vdn_per_seed = {
            seed: _metrics(vdn_errors[seed][mask], vdn_passed[seed][mask])
            for seed in EXPECTED_SEEDS
        }
        geoprr_per_seed = {
            seed: _metrics(geoprr_errors[seed][mask], geoprr_passed[seed][mask])
            for seed in EXPECTED_SEEDS
        }
        vdn_mean_error = np.mean(
            np.stack([vdn_errors[seed] for seed in EXPECTED_SEEDS]), axis=0
        )
        geoprr_mean_error = np.mean(
            np.stack([geoprr_errors[seed] for seed in EXPECTED_SEEDS]), axis=0
        )
        delta = geoprr_mean_error - vdn_mean_error
        vdn_nmae = float(np.mean(vdn_mean_error[mask]))
        geoprr_nmae = float(np.mean(geoprr_mean_error[mask]))
        bootstrap = _scene_bootstrap_delta(
            delta,
            scenes,
            mask,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + scope_index,
        )
        scope_reports[scope] = {
            "rows": int(np.sum(mask)),
            "samples": 1_558,
            "scenes": 14,
            "vdn": {
                "per_seed": {str(seed): vdn_per_seed[seed] for seed in EXPECTED_SEEDS},
                "across_seed": _across_seed_metrics(vdn_per_seed),
            },
            "geoprr": {
                "per_seed": {
                    str(seed): geoprr_per_seed[seed] for seed in EXPECTED_SEEDS
                },
                "across_seed": _across_seed_metrics(geoprr_per_seed),
            },
            "paired_scene_bootstrap": bootstrap,
            "relative_nmae_reduction_of_geoprr_vs_vdn": (
                float((vdn_nmae - geoprr_nmae) / vdn_nmae)
                if vdn_nmae > 0.0
                else None
            ),
            "geoprr_lower_nmae": bool(geoprr_nmae < vdn_nmae),
            "geoprr_advantage_ci_excludes_zero": bool(
                bootstrap["ci95_upper"] < 0.0
            ),
        }

    all_scope = scope_reports["all_conditions"]
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "comparison_role": (
            "matched open-source structured baseline with annotation-assisted "
            "direction-to-reading conversion"
        ),
        "identity": {
            "samples": 1_558,
            "conditions": list(CONDITIONS),
            "rows_per_seed": 9_348,
            "scene_clusters": 14,
            "training_seeds": list(EXPECTED_SEEDS),
            "checkpoints_per_method": 3,
            "same_sample_condition_pixels_verified": True,
        },
        "fairness": {
            "vdn_source_public": True,
            "vdn_direction_input": "conditioned canonical ROI pixels only",
            "vdn_offline_reading_conversion": (
                "SyncG annotated pivot and ordered scale endpoints"
            ),
            "vdn_deployable_end_to_end": False,
            "failure_policy": "failed VDN row receives normalized error 1.0",
            "checkpoint_selection": "fixed terminal epoch; no holdout selection",
            "outer_split_matched": True,
            "inner_split_scene_disjoint": True,
        },
        "sources": {
            "pixel_reference": str(Path(pixel_reference_path).resolve()),
            "vdn": {str(seed): vdn_sources[seed] for seed in EXPECTED_SEEDS},
            "geoprr": {str(seed): geoprr_sources[seed] for seed in EXPECTED_SEEDS},
        },
        "scopes": scope_reports,
        "claim_check": {
            "geoprr_mean_nmae_lower_than_vdn": bool(all_scope["geoprr_lower_nmae"]),
            "paired_scene_ci_supports_lower_geoprr_nmae": bool(
                all_scope["geoprr_advantage_ci_excludes_zero"]
            ),
            "per_condition_geoprr_mean_nmae_lower_than_vdn": {
                condition: bool(scope_reports[condition]["geoprr_lower_nmae"])
                for condition in CONDITIONS
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vdn", type=Path, action="append", required=True)
    parser.add_argument("--geoprr", type=Path, action="append", required=True)
    parser.add_argument("--pixel-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20262020)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = summarize(
        vdn_paths=args.vdn,
        geoprr_paths=args.geoprr,
        pixel_reference_path=args.pixel_reference,
        output_path=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    overall = report["scopes"]["all_conditions"]
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(Path(args.output).resolve()),
                "geoprr_nmae_percent_fs": overall["geoprr"]["across_seed"][
                    "nmae_percent_fs"
                ],
                "vdn_nmae_percent_fs": overall["vdn"]["across_seed"][
                    "nmae_percent_fs"
                ],
                "delta_ci_percent_fs": [
                    overall["paired_scene_bootstrap"]["ci95_lower_percent_fs"],
                    overall["paired_scene_bootstrap"]["ci95_upper_percent_fs"],
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONDITIONS", "EXPECTED_SEEDS", "PROTOCOL", "summarize"]
