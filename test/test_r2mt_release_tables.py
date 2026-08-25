from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "r2mt_net_tables.json"
INDUSTRIAL = ROOT / "results" / "r2mt_industrial_multimethod.json"
FIGURES = ROOT / "paper" / "r2mt_net_electronics_overleaf" / "figures"


def _rows(name: str) -> list[dict[str, str]]:
    with (FIGURES / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class R2MTReleaseTableTests(unittest.TestCase):
    def test_compact_tables_match_syncg_figure_source(self) -> None:
        tables = json.loads(TABLES.read_text(encoding="utf-8-sig"))
        rows = _rows("source_r2mt_main.csv")
        selected = {
            (row["metric"], row["method"]): row
            for row in rows
            if row["metric"] in {"all_conditions", "projective_pooled"}
        }
        for metric in ("all_conditions", "projective_pooled"):
            source = selected[(metric, "R2MT-Net")]
            released = tables["syncg"][metric]["r2mt_net"]
            self.assertAlmostEqual(released["mean"], 100.0 * float(source["mean_nmae"]), places=9)
            self.assertAlmostEqual(released["sample_sd"], 100.0 * float(source["sample_sd"]), places=9)

    def test_ocr_scope_is_accepted_outputs_only(self) -> None:
        tables = json.loads(TABLES.read_text(encoding="utf-8-sig"))
        ocr = tables["ocr_accepted_outputs"]
        self.assertEqual(ocr["accepted_labeled_frames"], 9)
        self.assertEqual(ocr["all_labeled_frames"], 33)
        self.assertIn("accepted labeled frames only", ocr["scope"])
        self.assertNotIn("full_denominator", ocr)
        self.assertNotIn("conditional_nmae", ocr)
        ocr_source = _rows("source_r2mt_ocr_accepted.csv")[0]
        self.assertNotIn("conditional_nmae", ocr_source)

    def test_compact_tables_match_four_model_industrial_summary(self) -> None:
        tables = json.loads(TABLES.read_text(encoding="utf-8-sig"))
        industrial = json.loads(INDUSTRIAL.read_text(encoding="utf-8-sig"))
        method_keys = {
            "R²MT-Net": "r2mt_net",
            "Direct-ResNet18": "direct_resnet18",
            "EfficientNet-B0": "efficientnet_b0",
            "MobileNetV3-Large": "mobilenet_v3_large",
        }
        for metric in ("all_conditions", "projective_pooled"):
            for display_name, machine_key in method_keys.items():
                source = industrial["metrics"][metric]["methods"][display_name]
                released = tables["industrial_roi"][metric][machine_key]
                self.assertAlmostEqual(
                    released["mean"], 100.0 * float(source["mean_nmae"]), places=8
                )
                self.assertAlmostEqual(
                    released["sample_sd"],
                    100.0 * float(source["sample_sd"]),
                    places=8,
                )


if __name__ == "__main__":
    unittest.main()
