"""Evaluate uncertainty arms on clean/25°/45° grouped validation only.

The separate main/mechanism decoder evidence also includes combined_severe;
that fourth condition is intentionally outside this secondary uncertainty
cohort.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)

from experiments.evaluate_pepd_grouped_validation import (
    GroupedValidationPerspectiveDataset,
    _group_bootstrap,
    _write_or_validate,
)
from experiments.pepd_convergence_protocol import (
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_SEEDS,
    GROUPED_VAL_BOOTSTRAP_ITERATIONS,
    GROUPED_VAL_BOOTSTRAP_SEED,
    GROUPED_VAL_DEGRADATION_SEED,
    PEPD_UNCERTAINTY_COHORT_PROTOCOL,
    PEPD_UNCERTAINTY_VERIFICATION_PROTOCOL,
    PROJECT_ROOT,
    UNCERTAINTY_MECHANISM_ARMS,
    UNCERTAINTY_GROUPED_VAL_CONDITIONS,
    formal_manifest_path,
    formal_parent_pin,
    sha256_file,
    uncertainty_output_dir,
)
from experiments.pepd_uncertainty_objectives import uncertainty_semantics
from experiments.pepd_uncertainty_metrics import (
    UNCERTAINTY_DIAGNOSTIC_PROTOCOL,
    grouped_uncertainty_bootstrap,
    grouped_uncertainty_diagnostics,
)
from experiments.probabilistic_pivot_direction import (
    build_probabilistic_pivot_direction_model,
    circular_delta,
    decode_probabilistic_pivot_direction,
)
from experiments.robustness_degradations import ROBUSTNESS_PROTOCOL
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_source_file,
)


EVALUATION_PROTOCOL = "pepd_uncertainty_grouped_val_perspective_v1"
COHORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "pepd_uncertainty_ablation_v1"
    / "cohort.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=UNCERTAINTY_MECHANISM_ARMS,
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument(
        "--condition",
        choices=UNCERTAINTY_GROUPED_VAL_CONDITIONS,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _direction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") is True]
    errors = np.asarray(
        [
            (
                float(row["angle_error_degrees"])
                if row.get("valid") is True
                else 180.0
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    pivot_valid = [
        row for row in rows if row.get("pivot_valid") is True
    ]
    pivots = np.asarray(
        [
            (
                float(row["pivot_error_fraction"])
                if row.get("pivot_valid") is True
                else math.sqrt(2.0)
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if not errors.size:
        raise ValueError("uncertainty evaluation has no samples")
    return {
        "samples": len(rows),
        "valid_directions": len(valid),
        "direction_coverage": len(valid) / len(rows),
        "invalid_direction_error_degrees": 180.0,
        "angle_mae_degrees": float(np.mean(errors)),
        "angle_median_degrees": float(np.median(errors)),
        "angle_acc_1deg": float(np.mean(errors <= 1.0)),
        "angle_acc_3deg": float(np.mean(errors <= 3.0)),
        "angle_acc_5deg": float(np.mean(errors <= 5.0)),
        "pivot_valid": len(pivot_valid),
        "pivot_coverage": len(pivot_valid) / len(rows),
        "invalid_pivot_error_fraction": math.sqrt(2.0),
        "pivot_mean_error_fraction": float(np.mean(pivots)),
        "pivot_median_error_fraction": float(np.median(pivots)),
        "successful_only_diagnostic": {
            "angle_mae_degrees": (
                None
                if not valid
                else float(
                    np.mean(
                        [
                            float(row["angle_error_degrees"])
                            for row in valid
                        ]
                    )
                )
            ),
            "pivot_mean_error_fraction": (
                None
                if not pivot_valid
                else float(
                    np.mean(
                        [
                            float(row["pivot_error_fraction"])
                            for row in pivot_valid
                        ]
                    )
                )
            ),
        },
        "primary_denominator_policy": (
            "all rows; invalid direction=180 degrees, invalid pivot=sqrt(2) "
            "image fraction, accuracy failures remain false"
        ),
    }


def run(args: argparse.Namespace) -> Path:
    arm = str(args.arm)
    seed = int(args.seed)
    condition = str(args.condition)
    if int(args.batch_size) != 64:
        raise ValueError("formal uncertainty evaluation batch size is fixed at 64")
    manifest = formal_manifest_path(args.manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train protocol hash drifted")
    cohort = _load(COHORT_PATH)
    if cohort.get("protocol") != PEPD_UNCERTAINTY_COHORT_PROTOCOL:
        raise ValueError("uncertainty cohort protocol mismatch")
    if cohort.get("all_runs_converged") is not True:
        raise ValueError("uncertainty cohort did not converge")
    if cohort.get("controlled_grouped_validation_authorized") is not True:
        raise ValueError("uncertainty grouped-validation is not authorized")
    if cohort.get("public_test_field_evaluation_authorized") is not False:
        raise ValueError("uncertainty cohort scope drifted")
    run_dir = uncertainty_output_dir(arm, seed)
    summary_path = run_dir / "summary.json"
    verification_path = run_dir / "verification.json"
    checkpoint_path = run_dir / "best.pt"
    summary = _load(summary_path)
    verification = _load(verification_path)
    if verification.get("protocol") != PEPD_UNCERTAINTY_VERIFICATION_PROTOCOL:
        raise ValueError("uncertainty verification protocol mismatch")
    if (
        verification.get("verified") is not True
        or verification.get("converged") is not True
    ):
        raise ValueError("uncertainty run is not verified/converged")
    if verification.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("uncertainty summary changed after verification")
    checkpoint_hash = sha256_file(checkpoint_path)
    if verification.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("uncertainty checkpoint changed after verification")
    signature = summary["signature"]
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    _, validation = grouped_train_val_split(
        samples,
        validation_fraction=float(signature["validation_fraction"]),
        seed=seed,
    )
    pin = formal_parent_pin(seed)
    if (
        len(validation) != pin.validation_samples
        or sample_ids_hash(validation) != pin.validation_ids_sha256
    ):
        raise ValueError("uncertainty grouped-validation identity drifted")
    dataset = GroupedValidationPerspectiveDataset(
        validation,
        image_size=int(signature["image_size"]),
        expansion=float(signature["expansion"]),
        condition=condition,
        degradation_seed=GROUPED_VAL_DEGRADATION_SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal uncertainty evaluation is CUDA-only")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_state = checkpoint.get("global_log_variance_state")
    if (
        not isinstance(global_state, torch.Tensor)
        or global_state.numel() != 1
        or not torch.isfinite(global_state).all()
    ):
        raise ValueError("uncertainty checkpoint global scalar is invalid")
    global_log_variance = global_state.to(
        device=device,
        dtype=torch.float32,
    ).reshape(())
    model = build_probabilistic_pivot_direction_model(
        angle_bins=int(signature["angle_bins"]),
        imagenet_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    rows: list[dict[str, Any]] = []
    stride = float(signature["image_size"]) / float(signature["heatmap_size"])
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_direction"].to(device, non_blocking=True)
            target_pivots = batch["target_pivot"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=True):
                outputs = model(images)
            prediction = decode_probabilistic_pivot_direction(
                *(value.float() for value in outputs)
            )
            semantics = uncertainty_semantics(
                outputs[3],
                mode=arm,
                global_log_variance_raw=global_log_variance,
            )
            predicted_angle = torch.atan2(
                prediction.direction[:, 1],
                prediction.direction[:, 0],
            )
            target_angle = torch.atan2(targets[:, 1], targets[:, 0])
            signed = circular_delta(predicted_angle, target_angle) * (
                180.0 / math.pi
            )
            error = torch.abs(signed)
            pivot_error = torch.linalg.vector_norm(
                prediction.pivot_xy * stride - target_pivots,
                dim=1,
            ) / float(signature["image_size"])
            pivot_valid = (
                torch.isfinite(prediction.pivot_xy).all(dim=1)
                & torch.isfinite(pivot_error)
            )
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "group_id": str(batch["group_id"][index]),
                        "valid": bool(prediction.valid[index].item()),
                        "angle_error_degrees": float(error[index].item()),
                        "signed_angle_error_degrees": float(signed[index].item()),
                        "angle_std_degrees": (
                            None
                            if semantics.angle_std_degrees is None
                            else float(
                                semantics.angle_std_degrees[index].item()
                            )
                        ),
                        "pivot_error_fraction": float(pivot_error[index].item()),
                        "pivot_valid": bool(pivot_valid[index].item()),
                        "calibration_semantics": semantics.calibration_semantics,
                    }
                )
    expected_ids = [sample.sample_id for sample in validation]
    actual_ids = [row["sample_id"] for row in rows]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("uncertainty evaluation output identity/order mismatch")
    output_path = run_dir / "grouped_validation" / f"{condition}.jsonl"
    payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _write_or_validate(output_path, payload)
    uncertainty_diagnostics = grouped_uncertainty_diagnostics(
        rows,
        mode=arm,
    )
    uncertainty_bootstrap = grouped_uncertainty_bootstrap(
        rows,
        mode=arm,
        iterations=GROUPED_VAL_BOOTSTRAP_ITERATIONS,
        seed=(
            GROUPED_VAL_BOOTSTRAP_SEED
            + UNCERTAINTY_GROUPED_VAL_CONDITIONS.index(condition)
        ),
    )
    report = {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "status": "complete",
        "scope": "SyncG official train grouped validation only",
        "role": "secondary mechanism ablation; not algorithm selection",
        "arm": arm,
        "seed": seed,
        "condition": condition,
        "condition_scope": (
            "uncertainty cohort only: clean, perspective_moderate, "
            "perspective_severe; combined_severe is not evaluated here"
        ),
        "degradation_protocol": ROBUSTNESS_PROTOCOL,
        "degradation_seed": GROUPED_VAL_DEGRADATION_SEED,
        "metrics": _direction_summary(rows),
        "grouped_metrics": _group_bootstrap(
            rows,
            iterations=GROUPED_VAL_BOOTSTRAP_ITERATIONS,
            seed=(
                GROUPED_VAL_BOOTSTRAP_SEED
                + UNCERTAINTY_GROUPED_VAL_CONDITIONS.index(condition)
            ),
        ),
        "calibration_semantics": rows[0]["calibration_semantics"],
        "calibration_claim_eligible": arm != "no_angular_nll",
        "group_uncertainty_ranking_claim_eligible": (
            arm == "learned_heteroscedastic"
        ),
        "uncertainty_diagnostic_protocol": (
            UNCERTAINTY_DIAGNOSTIC_PROTOCOL
        ),
        "uncertainty_diagnostics": uncertainty_diagnostics,
        "uncertainty_group_bootstrap": uncertainty_bootstrap,
        "checkpoint_sha256": checkpoint_hash,
        "verification_sha256": sha256_file(verification_path),
        "cohort_sha256": sha256_file(COHORT_PATH),
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "evaluator": sha256_source_file(Path(__file__).resolve()),
            "objective": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_uncertainty_objectives.py"
            ),
            "uncertainty_metrics": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "pepd_uncertainty_metrics.py"
            ),
        },
        "eligible_for_model_selection": False,
        "public_test_field_evaluation": False,
    }
    summary_output = output_path.with_suffix(".summary.json")
    report_payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _write_or_validate(summary_output, report_payload)
    print(report_payload, end="")
    print(summary_output)
    return summary_output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
