from __future__ import annotations

import unittest
from pathlib import Path

from experiments import formal_environment


class FormalEnvironmentTests(unittest.TestCase):
    def test_locked_distribution_versions_are_exact(self) -> None:
        locked = {
            "numpy": {"name": "numpy", "version": "1.26.4"},
            "torch": {"name": "torch", "version": "2.11.0+cu128"},
        }
        self.assertEqual(
            formal_environment.validate_locked_distributions(
                locked,
                {
                    "numpy": "1.26.4",
                    "torch": "2.11.0+cu128",
                    "unrelated": "9.9",
                },
            ),
            {
                "numpy": "1.26.4",
                "torch": "2.11.0+cu128",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "numpy"):
            formal_environment.validate_locked_distributions(
                locked,
                {
                    "numpy": "2.0.0",
                    "torch": "2.11.0+cu128",
                },
            )

    def test_current_formal_venv_matches_all_locked_distributions(
        self,
    ) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        identity = formal_environment.formal_environment_identity(
            project_dir,
            validate_process=False,
        )
        self.assertEqual(identity["python_version"], "3.11.15")
        self.assertEqual(
            identity["requirements_lock_sha256"],
            formal_environment.PINNED_REQUIREMENTS_TRAINING_LOCK_SHA256,
        )
        self.assertEqual(len(identity["locked_distributions"]), 82)


if __name__ == "__main__":
    unittest.main()
