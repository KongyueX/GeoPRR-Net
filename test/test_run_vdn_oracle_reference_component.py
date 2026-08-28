from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments import robustness_degradations
from experiments import run_cagh_v5_plain_paper_batch as primary_batch
from experiments import run_vdn_oracle_reference_component as oracle


class _FakeDirectionPredictor:
    native_input_size = 384

    def __init__(self, *, fail_condition: str | None = None) -> None:
        self.fail_condition = fail_condition
        self.requests: list[oracle.DirectionRequest] = []

    def predict_batch(self, requests):
        self.requests.extend(requests)
        records = []
        for request in requests:
            if request.condition == self.fail_condition:
                records.append(
                    {
                        "status": False,
                        "prediction_progress": None,
                        "failure_code": "synthetic_direction_failure",
                        "reference_input_sha256": oracle._reference_sha256(
                            oracle._DIRECTION_ONLY_REFERENCE
                        ),
                        "telemetry": {},
                    }
                )
            else:
                records.append(
                    {
                        # Mirrors the real adapter: its fixed non-GT dummy
                        # reference makes status false, while direction
                        # telemetry remains available for offline conversion.
                        "status": False,
                        "prediction_progress": None,
                        "failure_code": "direction_only_no_reference",
                        "reference_input_sha256": oracle._reference_sha256(
                            oracle._DIRECTION_ONLY_REFERENCE
                        ),
                        "telemetry": {"pointer_angle": 45.0},
                    }
                )
        return records


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(oracle._canonical_json_line(row) for row in rows))


def _fixture(tmp_path: Path, *, bbox=(10.0, 20.0, 110.0, 120.0)):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(100, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.arange(100, dtype=np.uint8)[:, None]
    image[:, :, 2] = 127
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    assert ok
    payload = encoded.tobytes()
    roi_path = tmp_path / "roi.png"
    roi_path.write_bytes(payload)
    roi_manifest = tmp_path / "roi_manifest.jsonl"
    _write_jsonl(
        roi_manifest,
        [
            {
                "sample_id": "s0",
                "roi_path": roi_path.name,
                "roi_png_sha256": hashlib.sha256(payload).hexdigest(),
                "roi_pixel_sha256": primary_batch.canonical_roi_pixel_sha256(image),
            }
        ],
    )
    source_manifest = tmp_path / "syncg.jsonl"
    _write_jsonl(
        source_manifest,
        [
            {
                "sample_id": "s0",
                "group_id": "g0",
                "ground_truth": 50.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
                "metadata": {
                    "dial_bbox": list(bbox),
                    "keypoints": [
                        {
                            "type": "ScaleMark",
                            "all_kp": [[60.0, 110.0], [10.0, 70.0]],
                        },
                        # No outside_kp is provided on purpose.  The oracle
                        # component may consume the pivot, never GT direction.
                        {"type": "Pointer", "origin_kp": [60.0, 70.0]},
                    ],
                },
            }
        ],
    )
    validation_ids = tmp_path / "validation_ids.json"
    validation_ids.write_text('["s0"]\n', encoding="utf-8")
    return image, roi_manifest, source_manifest, validation_ids


def _source_reference() -> oracle.SourceReference:
    return oracle.SourceReference(
        sample_id="s0",
        group_id="g0",
        bbox_xyxy=(10.0, 20.0, 110.0, 120.0),
        pivot_xy=(60.0, 70.0),
        start_xy=(60.0, 110.0),
        end_xy=(10.0, 70.0),
    )


def test_reference_geometry_uses_ordered_marks_and_transforms_perspective() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    source = _source_reference()
    clean_image, clean_metadata = robustness_degradations.apply_degradation(
        image, "clean", sample_id="s0", seed=oracle.ROBUSTNESS_SEED
    )
    clean = oracle.reference_packet_for_condition(
        source,
        roi_shape=clean_image.shape,
        degradation_metadata=clean_metadata,
        native_input_size=384,
    )
    assert clean["start_angle"] == pytest.approx(0.0)
    assert clean["range_angle"] == pytest.approx(90.0)

    blurred_image, blurred_metadata = robustness_degradations.apply_degradation(
        image, "blur_severe", sample_id="s0", seed=oracle.ROBUSTNESS_SEED
    )
    blurred = oracle.reference_packet_for_condition(
        source,
        roi_shape=blurred_image.shape,
        degradation_metadata=blurred_metadata,
        native_input_size=384,
    )
    assert blurred["start_angle"] == pytest.approx(clean["start_angle"])
    assert blurred["range_angle"] == pytest.approx(clean["range_angle"])

    warped_image, warped_metadata = robustness_degradations.apply_degradation(
        image, "perspective_severe", sample_id="s0", seed=oracle.ROBUSTNESS_SEED
    )
    warped = oracle.reference_packet_for_condition(
        source,
        roi_shape=warped_image.shape,
        degradation_metadata=warped_metadata,
        native_input_size=384,
    )
    assert warped["status"] is True
    assert 0.0 < warped["range_angle"] < 360.0
    assert (
        warped["start_angle"] != pytest.approx(clean["start_angle"])
        or warped["range_angle"] != pytest.approx(clean["range_angle"])
    )


def test_complete_six_condition_run_keeps_model_failure_and_is_oracle_only(
    tmp_path: Path,
) -> None:
    _image, roi_manifest, source_manifest, validation_ids = _fixture(tmp_path)
    fake = _FakeDirectionPredictor(fail_condition="perspective_moderate")
    predictions = tmp_path / "oracle_predictions.jsonl"
    count = oracle.run_component(
        roi_manifest_path=roi_manifest,
        syncg_manifest_path=source_manifest,
        output_path=predictions,
        batch_size=4,
        expected_samples=1,
        expected_groups=1,
        expected_ids_sha256=None,
        expected_roi_manifest_sha256=None,
        expected_source_manifest_sha256=None,
        predictor_factory=lambda *, device: fake,
    )
    rows = [json.loads(line) for line in predictions.read_text("utf-8").splitlines()]
    assert count == len(oracle.CONDITIONS) == 6
    assert len(fake.requests) == 6
    assert all(not hasattr(request, "reference") for request in fake.requests)
    assert [row["condition"] for row in rows] == list(oracle.CONDITIONS)
    assert all(set(row) == set(oracle.OUTPUT_KEYS) for row in rows)
    assert all(row["evaluation_role"] == oracle.EVALUATION_ROLE for row in rows)
    assert all(row["primary_table_eligible"] is False for row in rows)
    assert all(row["deployable"] is False for row in rows)
    assert all(row["used_as_runtime_input"] is False for row in rows)
    assert all(
        row["oracle_reference_used_for_offline_progress_conversion"] is True
        for row in rows
    )
    assert all("ground_truth" not in row and "scale_start" not in row for row in rows)
    failed = next(row for row in rows if row["condition"] == "perspective_moderate")
    assert failed["status"] == "fail"
    assert failed["normalized_progress"] is None
    assert failed["failure_code"] == "synthetic_direction_failure"
    clean = next(row for row in rows if row["condition"] == "clean")
    assert clean["normalized_progress"] == pytest.approx(0.5)

    result = oracle.score_component(
        predictions_path=predictions,
        syncg_manifest_path=source_manifest,
        validation_ids_path=validation_ids,
        expected_samples=1,
        expected_groups=1,
        bootstrap_replicates=5,
        bootstrap_seed=7,
    )
    assert result["protocol"] == oracle.SCORE_PROTOCOL
    assert result["primary_table_eligible"] is False
    assert result["full_three_seed_summary"] == []
    failure_cell = next(
        cell
        for cell in result["method_condition_results"]
        if cell["condition"] == "perspective_moderate"
    )
    assert failure_cell["samples"] == 1
    assert failure_cell["failures"] == 1
    assert failure_cell["metrics"]["coverage"] == 0.0
    assert failure_cell["metrics"]["nmae"] == 1.0


def test_bbox_geometry_mismatch_emits_all_failures_without_calling_model(
    tmp_path: Path,
) -> None:
    _image, roi_manifest, source_manifest, _validation_ids = _fixture(
        tmp_path, bbox=(10.0, 20.0, 109.0, 120.0)
    )
    fake = _FakeDirectionPredictor()
    predictions = tmp_path / "invalid_reference_predictions.jsonl"
    count = oracle.run_component(
        roi_manifest_path=roi_manifest,
        syncg_manifest_path=source_manifest,
        output_path=predictions,
        batch_size=3,
        expected_samples=1,
        expected_groups=1,
        expected_ids_sha256=None,
        expected_roi_manifest_sha256=None,
        expected_source_manifest_sha256=None,
        predictor_factory=lambda *, device: fake,
    )
    rows = [json.loads(line) for line in predictions.read_text("utf-8").splitlines()]
    assert count == 6
    assert fake.requests == []
    assert all(row["status"] == "fail" for row in rows)
    assert all(
        row["failure_code"].startswith("oracle_reference_invalid:") for row in rows
    )
    assert all(row["oracle_reference_sha256"] is None for row in rows)


def test_oracle_method_does_not_replace_native_primary_vdn_row() -> None:
    assert oracle.NATIVE_AUTO_REFERENCE_METHOD in primary_batch.METHODS
    assert oracle.METHOD not in primary_batch.METHODS
    assert len(primary_batch.METHODS) == 8
