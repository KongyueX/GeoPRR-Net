"""Small public/synthetic checks for the shared SyncG meter frontend."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.build_syncg_meter_detector_public import _normalized_box, verify_corpus
from experiments.field_blind_multimethod import select_shared_roi
from experiments.syncg_meter_detector_frontend import (
    EXPECTED_DETECTOR_SOURCE,
    _validate_production_source,
)
from experiments.train_syncg_meter_detector import choose_confidence_threshold


class _SyntheticDetector:
    def target_detection(self, image: np.ndarray, confidence: float):
        del image, confidence
        # The first item is intentionally lower-confidence; deterministic
        # selection must choose the second and apply symmetric 5% padding.
        return (
            [0.70, 0.90],
            [
                np.asarray([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32),
                np.asarray([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32),
            ],
            [np.zeros((40, 40, 3), dtype=np.uint8), np.zeros((60, 60, 3), dtype=np.uint8)],
            [0, 0],
        )


def main() -> None:
    normalized, line, clipped = _normalized_box(
        [-2, 20, 102, 80], width=100, height=100, sample_id="synthetic"
    )
    assert normalized == [0.5, 0.5, 1.0, 0.6]
    assert clipped == [0.0, 20.0, 100.0, 80.0]
    assert line.startswith("0 0.5000000000 0.5000000000")

    samples = [
        {"sample_id": "a", "dial_bbox_xyxy": [10, 10, 90, 90]},
        {"sample_id": "b", "dial_bbox_xyxy": [0, 0, 50, 50]},
    ]
    predictions = [
        {
            "sample_id": "a",
            "candidates": [{"confidence": 0.90, "xyxy": [10, 10, 90, 90]}],
        },
        {
            "sample_id": "b",
            "candidates": [{"confidence": 0.80, "xyxy": [0, 0, 50, 50]}],
        },
    ]
    choice = choose_confidence_threshold(samples, predictions)
    assert choice["branch"] == "recall_constraint_feasible"
    assert choice["selected"]["selected_box_iou50_recall"] == 1.0
    assert choice["selected"]["confidence_threshold"] == 0.80

    image = np.full((100, 100, 3), 127, dtype=np.uint8)
    roi, record = select_shared_roi(
        _SyntheticDetector(),
        image,
        confidence_threshold=0.50,
        padding_fraction=0.05,
        accepted_class_ids=[0],
    )
    assert roi is not None and roi.shape == (66, 66, 3)
    assert record["detected_xyxy"] == [20, 20, 80, 80]
    assert record["padded_xyxy"] == [17, 17, 83, 83]
    assert record["detector_invocations"] == 1
    assert record["fallback_to_full_frame"] is False

    _validate_production_source(EXPECTED_DETECTOR_SOURCE)
    summary = verify_corpus(Path(r"C:\pointer_read\syncg_meter_detector_public_v1"))
    assert summary["inventory"]["images"] == 16_000
    assert summary["inventory"]["physical_groups"] == 725
    assert summary["split"]["group_disjoint"] is True
    print(
        json.dumps(
            {
                "status": "passed",
                "synthetic_frontend": True,
                "public_corpus_verified": True,
                "images": summary["inventory"]["images"],
                "groups": summary["inventory"]["physical_groups"],
                "clipped_visible_boxes": summary["inventory"]["boxes_clipped_to_visible_frame"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
