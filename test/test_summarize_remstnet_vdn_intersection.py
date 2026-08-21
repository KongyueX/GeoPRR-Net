from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.summarize_remstnet_vdn_intersection import (
    CONDITIONS,
    EXPECTED_SOURCE_SEEDS,
    REMST_EVALUATION_PROTOCOL as EVALUATION_PROTOCOL,
    ReMSTNetVDNIntersectionError,
    summarize_vdn_intersection,
)


class ReMSTNetVDNIntersectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _external(target: float) -> dict[str, object]:
        predictions = [target + 0.03, target + 0.04, target + 0.05]
        return {
            "EfficientNet-B0": {
                "seeds": list(EXPECTED_SOURCE_SEEDS),
                "predictions": predictions,
                "absolute_errors": [abs(value - target) for value in predictions],
                "passed": [True, True, True],
            }
        }

    def _evaluation(self, seed: int) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        seed_index = EXPECTED_SOURCE_SEEDS.index(seed)
        for sample_index, scene in enumerate(("scene_a", "scene_b", "scene_c")):
            target = 0.25 + 0.2 * sample_index
            for condition_index, condition in enumerate(CONDITIONS):
                prediction = target + 0.01 + seed_index * 0.001
                rows.append(
                    {
                        "sample_id": f"sample_{sample_index}",
                        "scene_stem": scene,
                        "condition": condition,
                        "condition_pixel_sha256": (
                            f"pixel-{sample_index}-{condition_index}"
                        ),
                        "normalized_target": target,
                        "external": self._external(target),
                        "remstnet": {
                            "prediction": prediction,
                            "absolute_error": abs(prediction - target),
                        },
                    }
                )
        return {
            "protocol": EVALUATION_PROTOCOL,
            "status": "pilot_complete",
            "model": {"source_foundation": {"source_seed": seed}},
            "per_sample_condition": rows,
        }

    def _write_evaluations(self) -> list[Path]:
        paths: list[Path] = []
        for seed in EXPECTED_SOURCE_SEEDS:
            path = self.root / f"remst_{seed}.json"
            path.write_text(json.dumps(self._evaluation(seed)), encoding="utf-8")
            paths.append(path)
        return paths

    def _write_vdn(self, *, pixel_mismatch: bool = False) -> tuple[Path, Path]:
        automatic = self.root / "vdn_automatic.jsonl"
        oracle = self.root / "vdn_oracle.jsonl"
        automatic_rows: list[dict[str, object]] = []
        oracle_rows: list[dict[str, object]] = []
        for sample_index in range(2):
            target = 0.25 + 0.2 * sample_index
            for condition_index, condition in enumerate(CONDITIONS):
                pixel = f"pixel-{sample_index}-{condition_index}"
                if pixel_mismatch and sample_index == condition_index == 0:
                    pixel = "different-pixels"
                failed = sample_index == 1 and condition == "combined_severe"
                automatic_rows.append(
                    {
                        "protocol": "cagh_v5_plain_paper_batch_v1",
                        "method": "vdn_official200_terminal_seed20",
                        "sample_id": f"sample_{sample_index}",
                        "condition": condition,
                        "condition_pixel_sha256": pixel,
                        "status": "fail" if failed else "pass",
                        "normalized_progress": None if failed else target + 0.02,
                    }
                )
                oracle_rows.append(
                    {
                        "protocol": "vdn_syncg_oracle_reference_component_v1",
                        "method": (
                            "vdn_official200_terminal_seed20_"
                            "oracle_reference_component"
                        ),
                        "sample_id": f"sample_{sample_index}",
                        "condition": condition,
                        "condition_pixel_sha256": f"pixel-{sample_index}-{condition_index}",
                        "status": "pass",
                        "normalized_progress": target + 0.02,
                        "deployable": False,
                        "primary_table_eligible": False,
                        "oracle_reference_used_for_offline_progress_conversion": True,
                        "used_as_runtime_input": False,
                    }
                )
        automatic.write_text(
            "".join(json.dumps(row) + "\n" for row in automatic_rows),
            encoding="utf-8",
        )
        oracle.write_text(
            "".join(json.dumps(row) + "\n" for row in oracle_rows),
            encoding="utf-8",
        )
        return automatic, oracle

    def test_double_holdout_intersection_is_scored_without_merging_rosters(self) -> None:
        evaluations = self._write_evaluations()
        automatic, oracle = self._write_vdn()
        result = summarize_vdn_intersection(
            evaluations,
            automatic,
            oracle,
            bootstrap_replicates=20,
        )

        self.assertEqual(result["cohort"]["remstnet_holdout_samples"], 3)
        self.assertEqual(result["cohort"]["vdn_holdout_samples"], 2)
        self.assertEqual(result["cohort"]["intersection_samples"], 2)
        self.assertEqual(result["cohort"]["intersection_rows"], 12)
        all_conditions = result["summary"]["all_conditions"]
        self.assertAlmostEqual(
            all_conditions["remstnet_v3"]["metric_across_seed_mean_sd"][
                "nmae"
            ]["mean"],
            0.011,
        )
        self.assertEqual(
            all_conditions["vdn_official200_automatic_reference"]["failures"],
            1,
        )
        conditioned = all_conditions[
            "vdn_automatic_success_conditioned_comparison"
        ]
        self.assertEqual(conditioned["rows"], 11)
        self.assertEqual(conditioned["samples"], 2)
        self.assertAlmostEqual(
            conditioned["coverage_against_fixed_intersection"], 11 / 12
        )
        self.assertAlmostEqual(
            conditioned["remstnet_v3"]["metric_across_seed_mean_sd"][
                "nmae"
            ]["mean"],
            0.011,
        )
        self.assertAlmostEqual(
            conditioned["vdn_official200_automatic_reference"]["metrics"][
                "nmae"
            ],
            0.02,
        )
        self.assertFalse(
            all_conditions["vdn_official200_annotation_reference"]["deployable"]
        )

    def test_condition_pixel_mismatch_is_rejected(self) -> None:
        evaluations = self._write_evaluations()
        automatic, oracle = self._write_vdn(pixel_mismatch=True)
        with self.assertRaisesRegex(
            ReMSTNetVDNIntersectionError, "condition pixel identity differs"
        ):
            summarize_vdn_intersection(
                evaluations,
                automatic,
                oracle,
                bootstrap_replicates=2,
            )


if __name__ == "__main__":
    unittest.main()
