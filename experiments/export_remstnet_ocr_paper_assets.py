"""Export auditable source data and deterministic examples for the OCR deployment figure.

This utility consumes the scored end-to-end artifact rather than recomputing any
metric.  It writes compact CSV source data plus two unadjusted detector crops:
the median-error accepted labeled frame and the median sample identifier from
the most frequent labeled range-decoder failure category.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _median_row(rows: list[dict[str, Any]], key) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select a median row from an empty collection")
    ordered = sorted(rows, key=key)
    return ordered[(len(ordered) - 1) // 2]


def _mean_mapping_values(values: dict[str, float]) -> float:
    numbers = [float(value) for value in values.values()]
    if not numbers:
        raise ValueError("Expected at least one reader prediction")
    return sum(numbers) / len(numbers)


def _crop_detector_box(raw_root: Path, row: dict[str, Any], destination: Path) -> None:
    source = raw_root / f"{row['sample_id']}.png"
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not decode source image: {source}")
    x1, y1, x2, y2 = [int(value) for value in row["detector"]["bbox_xyxy"]]
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Detector box is outside {source}: {(x1, y1, x2, y2)}")
    crop = image[y1:y2, x1:x2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), crop):
        raise OSError(f"Could not write crop: {destination}")


def export_assets(score_path: Path, raw_root: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    coverage = payload["coverage_all_frames"]
    total = int(coverage["denominator"])
    if total != 153 or int(payload["samples"]["labeled_full_frames_for_accuracy"]) != 33:
        raise ValueError("Unexpected OCR deployment cohort size")

    coverage_rows = [
        {
            "stage": "detector",
            "display_label": "Cached detector",
            "count": int(coverage["detector"]),
            "denominator": total,
            "fraction": float(coverage["detector"]) / total,
        },
        {
            "stage": "ocr_token",
            "display_label": "Any numeric OCR token",
            "count": int(coverage["ocr_any_token"]),
            "denominator": total,
            "fraction": float(coverage["ocr_any_token"]) / total,
        },
        {
            "stage": "decoder_default",
            "display_label": "Physical reading (default)",
            "count": int(coverage["end_to_end_decoder_default"]),
            "denominator": total,
            "fraction": float(coverage["end_to_end_decoder_default"]) / total,
        },
        {
            "stage": "public_conservative",
            "display_label": "Physical reading (0.95 sensitivity)",
            "count": int(coverage["end_to_end_public_conservative"]),
            "denominator": total,
            "fraction": float(coverage["end_to_end_public_conservative"]) / total,
        },
    ]
    _write_csv(
        output_dir / "source_ocr_e2e_coverage.csv",
        ["stage", "display_label", "count", "denominator", "fraction"],
        coverage_rows,
    )

    accepted = [row for row in payload["per_image"] if bool(row["status_decoder_default"])]
    if len(accepted) != 9:
        raise ValueError(f"Expected nine accepted labeled frames, found {len(accepted)}")
    accepted_rows: list[dict[str, Any]] = []
    for row in sorted(accepted, key=lambda item: item["sample_id"]):
        accepted_rows.append(
            {
                "sample_id": row["sample_id"],
                "ground_truth_physical": float(row["ground_truth"]),
                "true_start": float(row["true_start"]),
                "true_end": float(row["true_end"]),
                "predicted_start": float(row["automatic_range"]["predicted_start"]),
                "predicted_end": float(row["automatic_range"]["predicted_end"]),
                "range_confidence": float(row["automatic_range"]["confidence"]),
                "remst_progress_mean": _mean_mapping_values(row["remst_progress_by_seed"]),
                "oracle_range_absolute_error_mean": float(row["three_seed_mean_errors"]["oracle_range"]),
                "automatic_range_absolute_error_mean": float(row["three_seed_mean_errors"]["decoder_default"]),
            }
        )
    _write_csv(
        output_dir / "source_ocr_e2e_accepted.csv",
        list(accepted_rows[0].keys()),
        accepted_rows,
    )

    range_metrics = payload["range_metrics_on_labeled_frames"]
    range_rows = [
        {
            "tolerance": "5% of true span",
            "numerator": 6,
            "denominator": 9,
            "fraction": float(range_metrics["pair_within_5_percent_true_span_conditional"]),
        },
        {
            "tolerance": "10% of true span",
            "numerator": 9,
            "denominator": 9,
            "fraction": float(range_metrics["pair_within_10_percent_true_span_conditional"]),
        },
    ]
    _write_csv(
        output_dir / "source_ocr_e2e_range.csv",
        ["tolerance", "numerator", "denominator", "fraction"],
        range_rows,
    )

    candidate_rows = []
    selected = payload["candidate_selection"]["selected_candidate"]
    for row in payload["candidate_selection"]["public_calibration"]:
        candidate_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "selected": str(row["candidate_id"] == selected).lower(),
                "samples": int(row["samples"]),
                "conditional_rounded_range_pair_accuracy": float(row["range_pair_rounded_exact_conditional"]),
                "full_denominator_rounded_range_pair_accuracy": float(row["range_pair_rounded_exact_full_denominator"]),
                "range_coverage": float(row["range_coverage"]),
                "mean_seconds_per_sample": float(row["mean_sample_seconds"]),
            }
        )
    _write_csv(
        output_dir / "source_ocr_candidate_screen.csv",
        list(candidate_rows[0].keys()),
        candidate_rows,
    )

    accepted_example = _median_row(
        accepted,
        key=lambda row: (float(row["three_seed_mean_errors"]["decoder_default"]), row["sample_id"]),
    )
    failures = [
        row
        for row in payload["per_image"]
        if not bool(row["status_decoder_default"])
        and row["automatic_range"].get("failure_reason")
    ]
    failure_counts = Counter(str(row["automatic_range"]["failure_reason"]) for row in failures)
    dominant_failure = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    failure_example = _median_row(
        [row for row in failures if row["automatic_range"]["failure_reason"] == dominant_failure],
        key=lambda row: row["sample_id"],
    )
    example_specs = [
        (
            "accepted_median_error",
            "median automatic-range absolute error among nine accepted labeled frames",
            accepted_example,
            "source_examples/ocr_e2e_accepted_median.png",
        ),
        (
            "dominant_failure_median_id",
            "median sample identifier within the most frequent labeled decoder failure category",
            failure_example,
            "source_examples/ocr_e2e_dominant_failure.png",
        ),
    ]
    example_rows = []
    for role, rule, row, relative_name in example_specs:
        destination = output_dir / relative_name
        _crop_detector_box(raw_root, row, destination)
        x1, y1, x2, y2 = [int(value) for value in row["detector"]["bbox_xyxy"]]
        example_rows.append(
            {
                "selection_role": role,
                "selection_rule": rule,
                "sample_id": row["sample_id"],
                "image_file": relative_name,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "ground_truth_physical": float(row["ground_truth"]),
                "true_start": float(row["true_start"]),
                "true_end": float(row["true_end"]),
                "predicted_start": row["automatic_range"].get("predicted_start"),
                "predicted_end": row["automatic_range"].get("predicted_end"),
                "range_confidence": float(row["automatic_range"]["confidence"]),
                "automatic_range_absolute_error_mean": float(row["three_seed_mean_errors"]["decoder_default"]),
                "failure_reason": row["automatic_range"].get("failure_reason") or "",
                "image_adjustment": "detector crop only; no contrast, brightness, gamma, or local editing",
            }
        )
    _write_csv(
        output_dir / "source_ocr_e2e_examples.csv",
        list(example_rows[0].keys()),
        example_rows,
    )

    return {
        "coverage_rows": len(coverage_rows),
        "accepted_rows": len(accepted_rows),
        "range_rows": len(range_rows),
        "candidate_rows": len(candidate_rows),
        "example_rows": len(example_rows),
        "accepted_example": accepted_example["sample_id"],
        "failure_example": failure_example["sample_id"],
        "dominant_failure": dominant_failure,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_assets(args.score, args.raw_root, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
