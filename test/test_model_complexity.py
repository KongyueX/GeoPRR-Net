import unittest

import numpy as np

from experiments.benchmark_model_complexity import (
    _latency_metrics,
    _tree_nodes,
    _unique_parameter_counts,
    validate_payload,
)


class _Tree:
    def __init__(self, nodes):
        self.tree_ = type("TreeState", (), {"node_count": nodes})()


class _Forest:
    def __init__(self, nodes):
        self.estimators_ = [_Tree(value) for value in nodes]


class ModelComplexityTests(unittest.TestCase):
    def test_parameter_count_deduplicates_shared_modules(self):
        import torch

        model = torch.nn.Linear(3, 2)
        counts = _unique_parameter_counts([model, model])
        self.assertEqual(counts["total"], 8)
        self.assertEqual(counts["trainable"], 8)

    def test_tree_node_count_is_recursive(self):
        self.assertEqual(_tree_nodes(_Forest([3, 7, 11])), 21)

    def test_latency_metrics_use_full_population(self):
        result = _latency_metrics([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["iterations"], 4)
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["serial_fps_from_mean"], 400.0)
        self.assertTrue(np.isfinite(result["p95_ms"]))

    def test_payload_validator_recomputes_totals(self):
        methods = []
        for label in ("Original Transformer", "VDN", "Ours-final"):
            methods.append(
                {
                    "label": label,
                    "parameters": {"total": 5},
                    "flops": 7,
                    "peak_gpu_allocated_bytes": 50,
                    "components": {
                        "component": {
                            "parameters": 5,
                            "flops_per_sample": 7,
                        }
                    },
                    "latency": {
                        "neural_stack": {
                            "iterations": 10,
                            "mean_ms": 1.0,
                            "p95_ms": 1.2,
                        },
                        "total_active_stack": {
                            "iterations": 10,
                            "mean_ms": 1.1,
                            "p95_ms": 1.3,
                        },
                        "cpu_postprocess": (
                            {"iterations": 10, "mean_ms": 0.1, "p95_ms": 0.2}
                            if label == "Ours-final"
                            else None
                        ),
                    },
                }
            )
        result = validate_payload(
            {
                "methods": methods,
                "timed_iterations": 10,
                "gpu_total_memory_bytes": 100,
            }
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["checks"], 16)


if __name__ == "__main__":
    unittest.main()
