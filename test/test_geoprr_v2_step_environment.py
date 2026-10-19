"""Keep the rendering subprocess environment separate from the reader process."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_geoprr_v2_optimization import OptimizationRun


class StepEnvironmentTests(unittest.TestCase):
    def test_only_child_receives_render_environment(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, GEOPRR_PROBE_TEST="reader"):
            root = Path(temporary)
            result = root / "observed.json"
            run = OptimizationRun(root, root / "queue", [20262020])
            command = [sys.executable, "-c",
                       "import os,json,sys;from pathlib import Path;"
                       "Path(sys.argv[1]).write_text(json.dumps({'value':os.environ['GEOPRR_PROBE_TEST']}))",
                       str(result)]
            run.step("render_environment", command, result, environment_overrides={"GEOPRR_PROBE_TEST": "renderer"})
            self.assertEqual(json.loads(result.read_text())["value"], "renderer")
            self.assertEqual(os.environ["GEOPRR_PROBE_TEST"], "reader")
            self.assertEqual(run.state["steps"]["render_environment"]["environment_overrides"],
                             {"GEOPRR_PROBE_TEST": "renderer"})


if __name__ == "__main__":
    unittest.main()
