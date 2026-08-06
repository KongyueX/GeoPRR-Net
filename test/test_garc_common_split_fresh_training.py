from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.automatic_numeric_range_public_protocol import sha256_file, strict_json
from experiments import garc_common_split_fresh_training as fresh
from experiments import garc_common_split_progress as common


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PWSh = Path(r"C:\Program Files\PowerShell\7\pwsh.exe")
WRAPPER = PROJECT_ROOT / "experiments/run_garc_common_split_fresh_training_event_driven.ps1"


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def complete_spec(protocol_file: Path, protocol: dict) -> dict:
    initialization = (
        Path(fresh.torch.hub.get_dir())
        / "checkpoints"
        / Path(fresh.IMAGENET_WEIGHTS.url).name
    ).resolve(strict=True)
    augmentation = fresh.v5_base.PhotoAugmentation.disabled().__dict__
    return {
        "schema_version": 1,
        "protocol": fresh.EXECUTION_SPEC_PROTOCOL,
        "status": "frozen_before_common_split_training",
        "parent_protocol": {
            "path": str(protocol_file),
            "sha256": sha256_file(protocol_file),
        },
        "seeds": list(common.EXPECTED_SEEDS),
        "output_root": str(fresh.DEFAULT_OUTPUT_ROOT),
        "source_bindings": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in fresh.SOURCE_FILES.items()
        },
        "partitions": protocol["partitions"],
        "pepd": {
            "imagenet_initialization": {
                "path": str(initialization),
                "sha256": sha256_file(initialization),
            },
            "phase1_epochs": 30,
            "phase2_epochs": 30,
            "batch_size": 24,
            "workers": 0,
            "image_size": 256,
            "heatmap_size": 64,
            "angle_bins": 72,
            "learning_rate": 3e-4,
            "phase1_eta_min": 3e-6,
            "phase2_learning_rate": 3e-6,
            "weight_decay": 1e-4,
            "pivot_loss_weight": 1.0,
            "bin_loss_weight": 0.2,
            "vector_loss_weight": 0.5,
            "paired_supervision_weight": 1.0,
            "equivariance_weight": 0.5,
            "equivariance_pivot_weight": 1.0,
            "soft_target_sigma_bins": 1.25,
            "expansion": 1.25,
            "scale_factor": 0.1,
            "rotation_factor": 90.0,
            "translation_factor": 0.12,
            "heatmap_sigma": 1.5,
            "perspective_probability": 0.8,
            "max_perspective_degrees": 45.0,
            "max_blur_sigma": 3.0,
            "intermediate_selection_metric": "calibration_angle_mae_then_nll_then_pivot_error",
        },
        "v5_enhanced": {
            "stage_a_epochs": 6,
            "stage_b_epochs": 12,
            "batch_size": 16,
            "workers": 0,
            "learning_rate": 4e-4,
            "weight_decay": 1e-4,
            "augmentation": augmentation,
            "checkpoint_selection_metric": "minimum_calibration_progress_mae_failure_penalty_1_then_coverage",
        },
        "progress_fusion": {
            "family": "ridge_residual_plus_confidence_abstention",
            "fit_partition": "algorithm_fit",
            "selection_partition": "calibration",
            "ridge_lambdas": [0.001, 0.01],
            "residual_scales": [0.0, 0.5, 1.0],
            "confidence_thresholds": [0.0, 0.25, 0.5],
            "mask_geometry_policy": "omitted_no_eligible_algorithm_fit_only_binding",
            "mask_geometry_binding": None,
        },
    }


def fusion_row(index: int, *, target_shift: float = 0.02) -> dict:
    prediction = 0.1 + 0.05 * index
    return {
        "status": "ok",
        "predicted_progress": prediction,
        "target_progress": prediction + target_shift,
        "endpoint": {
            "peak": [0.9, 0.8],
            "entropy": [0.1, 0.2],
            "separation": 0.5,
            "coordinate_disagreement": 0.01,
            "js_divergence": 0.02,
            "consistency_reliability": 0.9,
        },
        "arc": {"arc_length_reliability": 0.9},
        "reliability": {
            "gate_confidence": 0.8,
            "uncertainty_score": 0.1,
            "radius_reliability": 0.9,
            "endpoint_tick_support": 0.8,
        },
    }


class FreshCommonSplitTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_file, cls.protocol = common.load_protocol()

    def test_current_common_protocol_has_four_machine_visible_execution_gaps(self) -> None:
        gaps = fresh.execution_spec_gaps(self.protocol)
        self.assertEqual(len(gaps), 4)
        self.assertTrue(any("fusion_candidate_grid" in gap for gap in gaps))
        self.assertTrue(any("checkpoint_selection" in gap for gap in gaps))

    def test_static_preflight_is_blocked_and_opens_no_data(self) -> None:
        with tempfile.TemporaryDirectory(prefix="garc_fresh_preflight_") as directory:
            output = Path(directory) / "preflight.json"
            fresh.preflight(protocol_path=self.protocol_file, output_path=output)
            result = strict_json(output)
        self.assertFalse(result["training_allowed"])
        self.assertEqual(result["audit"]["images_opened"], 0)
        self.assertEqual(result["audit"]["annotations_opened"], 0)
        self.assertFalse(result["audit"]["gpu_initialized"])
        self.assertFalse(result["audit"]["process_wait_started"])

    def test_complete_separate_execution_spec_authenticates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="garc_fresh_spec_") as directory:
            spec_path = write_json(
                Path(directory) / "execution.json",
                complete_spec(self.protocol_file, self.protocol),
            )
            observed_path, observed = fresh.load_execution_spec(
                spec_path,
                protocol_file=self.protocol_file,
                protocol=self.protocol,
            )
        self.assertEqual(observed_path, spec_path.resolve())
        self.assertEqual(observed["seeds"], list(common.EXPECTED_SEEDS))

    def test_incomplete_fusion_grid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="garc_fresh_spec_") as directory:
            value = complete_spec(self.protocol_file, self.protocol)
            del value["progress_fusion"]["confidence_thresholds"]
            spec_path = write_json(Path(directory) / "execution.json", value)
            with self.assertRaisesRegex(ValueError, "confidence_thresholds"):
                fresh.load_execution_spec(
                    spec_path,
                    protocol_file=self.protocol_file,
                    protocol=self.protocol,
                )

    def test_progress_fusion_is_fit_on_fit_and_selected_on_calibration(self) -> None:
        fit = [fusion_row(index) for index in range(10)]
        calibration = [fusion_row(index, target_shift=0.01) for index in range(6)]
        config = complete_spec(self.protocol_file, self.protocol)["progress_fusion"]
        with tempfile.TemporaryDirectory(
            prefix="garc_fresh_fusion_", dir=r"C:\pointer_read"
        ) as directory:
            result = fresh.fit_and_select_progress_fusion(
                fit_rows=fit,
                calibration_rows=calibration,
                config=config,
                seed=20260816,
                output=Path(directory),
            )
            state = strict_json(Path(result["state"]["path"]))
        self.assertEqual(state["status"], "fit_on_algorithm_fit_selected_on_calibration")
        self.assertEqual(state["fit"]["samples"], 10)
        self.assertEqual(state["calibration"]["gradient_updates"], 0)
        self.assertGreater(state["calibration"]["selection_queries"], 0)

    def test_wrapper_is_event_driven_and_does_not_launch_in_preflight(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8-sig")
        self.assertIn("WaitForExit", source)
        self.assertNotIn("Start-Sleep", source)
        self.assertNotIn("Start-Process", source)
        self.assertIn("PYTHONHASHSEED", source)
        with tempfile.TemporaryDirectory(prefix="garc_fresh_wrapper_") as directory:
            output = Path(directory) / "preflight.json"
            completed = subprocess.run(
                [
                    str(PWSh),
                    "-NoLogo",
                    "-NoProfile",
                    "-File",
                    str(WRAPPER),
                    "-PreflightOnly",
                    "-PreflightOutput",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(strict_json(output)["training_allowed"])


if __name__ == "__main__":
    unittest.main()
