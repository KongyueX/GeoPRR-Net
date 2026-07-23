"""Tests for paper latency and failure reporting utilities."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.make_latency_table import render_markdown, summarize_cache


class ExperimentReportingTest(unittest.TestCase):
    def test_latency_summary_keeps_failures_and_early_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [
                {
                    "sample_id": "ok",
                    "status": True,
                    "runtime_seconds": 0.10,
                },
                {
                    "sample_id": "pointer",
                    "status": False,
                    "error_code": "pointer_not_found",
                    "runtime_seconds": 0.20,
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            path.with_name(path.name + ".meta.json").write_text(
                json.dumps(
                    {
                        "signature": {
                            "device": "cuda",
                            "reading_backend": "compare",
                            "include_transformer": True,
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_cache("test", path)
            self.assertEqual(summary["batch_size"], 1)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["failures"], 1)
            self.assertAlmostEqual(summary["all"]["mean_ms"], 150.0)
            self.assertEqual(
                summary["by_outcome"]["pointer_not_found"]["samples"],
                1,
            )
            markdown = render_markdown([summary])
            self.assertIn("| test | cuda | 2 | 150.00", markdown)
            self.assertIn("| test | Pointer not found | 1 | 200.00", markdown)

    def test_latency_summary_rejects_incomplete_pipeline_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(
                '{"status":true,"runtime_seconds":0.1}\n',
                encoding="utf-8",
            )
            path.with_name(path.name + ".meta.json").write_text(
                json.dumps(
                    {
                        "signature": {
                            "device": "cuda",
                            "reading_backend": "compare",
                            "include_transformer": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "complete reading pipeline"):
                summarize_cache("test", path)


if __name__ == "__main__":
    unittest.main()
