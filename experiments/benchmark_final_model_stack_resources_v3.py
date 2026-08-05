"""Corrected module entry point for the frozen same-roster resource benchmark.

The v2 direct-file entry failed before model construction because Python placed
``experiments/`` rather than the repository root on ``sys.path``.  This wrapper
only establishes that import root and assigns independent v3 protocol/output
identities; the bound v2 benchmark implementation is otherwise unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiments import benchmark_final_model_stack_resources as core  # noqa: E402


core.PROTOCOL = "paper_final_model_stack_same_roster_resources_v3"
core.FREEZE_PROTOCOL = "paper_final_model_stack_same_roster_resources_freeze_v3"
core.DEFAULT_FREEZE = (
    PROJECT_DIR
    / "artifacts"
    / "protocols"
    / "final_model_stack_same_roster_resources_v3_freeze.json"
)
core.DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "runs"
    / "efficiency"
    / "final_model_stack_same_roster_resources_v3"
)


if __name__ == "__main__":
    core.main()
