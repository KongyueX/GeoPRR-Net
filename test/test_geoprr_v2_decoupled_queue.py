"""Source-validation continuation for the separate Q/U training candidate."""
import unittest

from experiments.run_geoprr_v2_decoupled import validation_decision


class DecoupledDecisionTests(unittest.TestCase):
    def test_mean_improvement_cannot_hide_more_severe_errors(self):
        baseline = {"smoke_only": False, "summary": {"samples": 1506, "nmae": .002, "errors_above_5_percent_fs": 5, "tail10_nmae": .012}}
        for error, severe, expected in ((.0018, 5, "promising_source_validation"),
                                        (.0018, 6, "validation_rejected"),
                                        (.002, 4, "validation_rejected"),
                                        (.00198, 5, "validation_rejected")):
            candidate = {"smoke_only": False, "summary": {"samples": 1506, "nmae": error, "errors_above_5_percent_fs": severe, "tail10_nmae": .011}}
            result = validation_decision(candidate, baseline)
            self.assertEqual(result["decision"], expected)
            self.assertFalse(result["test_used_for_decision"])
        candidate["summary"].update(nmae=.0018, tail10_nmae=.013)
        self.assertEqual(validation_decision(candidate, baseline)["decision"], "validation_rejected")

    def test_incomplete_evaluation_cannot_decide(self):
        baseline = {"smoke_only": False, "summary": {"samples": 1506, "nmae": .002, "errors_above_5_percent_fs": 5, "tail10_nmae": .012}}
        for candidate in (dict(baseline, smoke_only=True), dict(baseline, summary=dict(baseline["summary"], samples=64))):
            with self.assertRaises(ValueError):
                validation_decision(candidate, baseline)


if __name__ == "__main__":
    unittest.main()
