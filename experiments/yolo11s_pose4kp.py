"""Dataset adapter for a YOLO11s-Pose model with four gauge keypoints."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from experiments.roi_geometry_comparison import (
    KEYPOINT_NAMES,
    canonical_roi_bounds,
    extract_syncg_keypoints,
    write_json,
)


METHOD = "YOLO11s-Pose-4KP"


def _materialize_split(
    samples: Sequence[Any],
    *,
    root: Path,
    split: str,
    image_size: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    invisible_points = 0
    for sample in samples:
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if image is None:
            raise ValueError(f"{sample.sample_id}: source image decode failed")
        left, top, right, bottom = canonical_roi_bounds(image.shape, sample.dial_bbox)
        roi = image[top:bottom, left:right]
        resized = cv2.resize(
            roi, (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )
        image_path = image_dir / f"{sample.sample_id}.jpg"
        if not cv2.imwrite(
            str(image_path), resized, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        ):
            raise OSError(f"{sample.sample_id}: failed to write YOLO image")
        points = extract_syncg_keypoints(sample) - np.asarray(
            [left, top], dtype=np.float32
        )
        normalizer = np.asarray([right - left, bottom - top], dtype=np.float32)
        normalized = points / normalizer[None, :]
        keypoint_fields: list[str] = []
        for x, y in normalized:
            visible = bool(-0.01 <= float(x) <= 1.01 and -0.01 <= float(y) <= 1.01)
            if visible:
                keypoint_fields.extend(
                    (f"{float(np.clip(x, 0.0, 1.0)):.8f}", f"{float(np.clip(y, 0.0, 1.0)):.8f}", "2")
                )
            else:
                invisible_points += 1
                keypoint_fields.extend(("0", "0", "0"))
        label = " ".join(("0", "0.5", "0.5", "1.0", "1.0", *keypoint_fields))
        (label_dir / f"{sample.sample_id}.txt").write_text(label + "\n", encoding="utf-8")
        image_paths.append(image_path.resolve())
    list_path = root / f"{split}.txt"
    list_path.write_text(
        "\n".join(path.as_posix() for path in image_paths) + "\n", encoding="utf-8"
    )
    return {
        "samples": len(image_paths),
        "list": str(list_path.resolve()),
        "invisible_keypoints": invisible_points,
    }


def materialize_pose_dataset(
    train_samples: Sequence[Any],
    validation_samples: Sequence[Any],
    *,
    output_root: Path,
    image_size: int,
    jpeg_quality: int = 95,
) -> tuple[Path, dict[str, Any]]:
    """Write one-gauge ROI images and Ultralytics four-keypoint labels."""

    root = Path(output_root).resolve()
    if (root / "data.yaml").exists():
        raise FileExistsError(f"YOLO pose dataset already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    train_summary = _materialize_split(
        train_samples,
        root=root,
        split="train",
        image_size=image_size,
        jpeg_quality=jpeg_quality,
    )
    validation_summary = _materialize_split(
        validation_samples,
        root=root,
        split="val",
        image_size=image_size,
        jpeg_quality=jpeg_quality,
    )
    yaml_path = root / "data.yaml"
    yaml_payload: Mapping[str, Any] = {
        "path": str(root),
        "train": "train.txt",
        "val": "val.txt",
        "names": {0: "gauge"},
        "kpt_shape": [len(KEYPOINT_NAMES), 3],
        "flip_idx": list(range(len(KEYPOINT_NAMES))),
        "kpt_names": {0: list(KEYPOINT_NAMES)},
    }
    yaml_path.write_text(
        yaml.safe_dump(dict(yaml_payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "method": METHOD,
        "image_size": int(image_size),
        "image_encoding": f"JPEG quality {int(jpeg_quality)}",
        "keypoint_order": list(KEYPOINT_NAMES),
        "object_box": "complete canonical ROI",
        "train": train_summary,
        "validation": validation_summary,
        "data_yaml": str(yaml_path.resolve()),
    }
    write_json(root / "materialization_summary.json", summary)
    return yaml_path, summary


__all__ = ["METHOD", "materialize_pose_dataset"]
