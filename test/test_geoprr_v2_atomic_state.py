from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_geoprr_v2_optimization import write_json


class AtomicQueueStateTests(unittest.TestCase):
    def test_reader_sees_complete_old_state_during_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queue_state.json"
            before = {"status": "running", "completed_groups": 170}
            after = {"status": "running", "completed_groups": 171}
            write_json(path, before)

            def partial_write(target: Path, text: str) -> None:
                with target.open("w") as handle:
                    midpoint = len(text) // 2
                    handle.write(text[:midpoint])
                    handle.flush()
                    self.assertEqual(json.loads(path.read_text()), before)
                    handle.write(text[midpoint:])

            with patch.object(Path, "write_text", partial_write):
                write_json(path, after)
            self.assertEqual(json.loads(path.read_text()), after)
            self.assertFalse(path.with_name("queue_state.json.tmp").exists())

    def test_interrupted_write_preserves_published_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queue_state.json"
            before = {"status": "running", "completed_groups": 170}
            write_json(path, before)

            def interrupted_write(target: Path, _text: str) -> None:
                with target.open("w") as handle:
                    handle.write('{"status":')
                raise OSError("interrupted write")

            with patch.object(Path, "write_text", interrupted_write):
                with self.assertRaises(OSError):
                    write_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), before)
            write_json(path, {"status": "complete"})
            self.assertEqual(json.loads(path.read_text()), {"status": "complete"})


if __name__ == "__main__":
    unittest.main()
