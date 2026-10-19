from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.run_geoprr_v2_optimization import OptimizationRun


class CompletedChildTests(unittest.TestCase):
    def test_completed_render_is_adopted_after_controller_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            run = OptimizationRun(home, home / "output", [20262020])
            expected = home / "group.json"
            expected.write_text('{"images": [1,2,3,4]}')
            run.state["steps"]["render"] = {"status": "running", "pid": 12345,
                                             "command": ["blender"], "expected": str(expected)}
            with patch("experiments.run_geoprr_v2_optimization.process_running", return_value=False), \
                    patch("experiments.run_geoprr_v2_optimization.subprocess.Popen") as launch:
                run.step("render", ["blender"], expected)
            launch.assert_not_called()
            self.assertEqual(run.state["steps"]["render"]["status"], "complete")
            self.assertIsNone(run.state["steps"]["render"]["exit_code"])
            self.assertEqual(expected.read_text(), '{"images": [1,2,3,4]}')


if __name__ == "__main__":
    unittest.main()
