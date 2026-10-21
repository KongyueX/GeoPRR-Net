"""Inspect the frozen feedback best48 checkpoint without optimization."""
from experiments.diagnose_geoprr_v2_reference_relation import main as diagnose


FOCUS = {"sync_2810", "sync_3663", "sync_10304", "sync_3172"}


def main(argv=None):
    return diagnose(argv,
        fixedreader_relative="runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020",
        expected_epoch=48, focus=FOCUS, capture_reference_feedback=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
