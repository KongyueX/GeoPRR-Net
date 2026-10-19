"""Prepare fixed scene-disjoint manifests after the SyncG upload is complete."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from experiments.datasets import SYNCG_SAMPLE_IDS_SHA256, SYNCG_TRAIN_ROWS, SYNCG_TEST_ROWS, sample_ids_sha256
from experiments.geoprr_v2_data import normalized_target


def syncg_row(annotation_path: Path, image_dir: Path) -> dict[str, Any]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    sample_id = annotation_path.stem
    file_name = str(annotation.get("file_name") or sample_id)
    candidates = [image_dir / file_name]
    if not Path(file_name).suffix:
        candidates += [image_dir / f"{file_name}{suffix}" for suffix in (".jpg", ".png", ".jpeg")]
    image_path = next((path for path in candidates if path.is_file()), None)
    if image_path is None:
        raise FileNotFoundError(f"{sample_id}: corresponding image has not finished uploading")
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        with image_path.open("rb") as handle:
            if image_path.stat().st_size < 2:
                raise ValueError(f"{sample_id}: empty or incomplete JPEG upload")
            handle.seek(-2, 2)
            if handle.read(2) != b"\xff\xd9":
                raise ValueError(f"{sample_id}: JPEG upload lacks its end marker")
    scene = str(annotation["scene_name"])
    gauge_type = str(annotation["gauge_type"])
    start = float(annotation["start_value"])
    interval = float(annotation["long_interval_value"])
    count = int(annotation["long_num"])
    row = {"dataset": "SyncG", "split": annotation_path.parent.name, "sample_id": sample_id,
           "group_id": f"{gauge_type}::{Path(scene).stem}", "meter_id": gauge_type,
           "image_path": str(image_path.resolve()), "ground_truth": float(annotation["ground_truth"]),
           "scale_start": start, "scale_end": start + (count - 1) * interval,
           "metadata": {"annotation_path": str(annotation_path.resolve()), "scene_name": scene,
                        "gauge_type": gauge_type, "pointer_angle": annotation["pointer_rotate_degree"],
                        "long_interval_degree": annotation["long_interval_degree"],
                        "long_interval_value": interval, "long_num": count,
                        "image_width": int(annotation["width"]), "image_height": int(annotation["height"]),
                        "dial_bbox": annotation["dial_bbox_annotations"],
                        "keypoints": annotation["keypoints_annotations"],
                        "homography": annotation["homo_matrix"]}}
    for key in ("homography_bbox", "layout_id", "factors"):
        if key in annotation:
            row["metadata"][key] = annotation[key]
    normalized_target(row)
    return row


def _scene(row: dict[str, Any]) -> str:
    return Path(row["metadata"]["scene_name"]).stem


def make_split(rows: list[dict[str, Any]], *, seed: int = 20261019,
               validation_fraction: float = 0.1, existing_split: Path | None = None) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    if existing_split is not None:
        split = json.loads(existing_split.read_text(encoding="utf-8-sig"))
        train_ids, validation_ids = split["train_sample_ids"], split["validation_sample_ids"]
    else:
        scenes = sorted({_scene(row) for row in rows})
        if len(scenes) < 2 or not 0 < validation_fraction < 1:
            raise ValueError("a scene-disjoint validation split needs at least two scenes and a fraction in (0,1)")
        count = max(1, min(len(scenes) - 1, int(len(scenes) * validation_fraction)))
        validation_scenes = set(random.Random(seed).sample(scenes, count))
        train_ids = sorted(row["sample_id"] for row in rows if _scene(row) not in validation_scenes)
        validation_ids = sorted(row["sample_id"] for row in rows if _scene(row) in validation_scenes)
    train, validation = set(train_ids), set(validation_ids)
    if len(train) != len(train_ids) or len(validation) != len(validation_ids) or train & validation or train | validation != set(by_id):
        raise ValueError("fixed split does not partition the source training sample IDs")
    train_scenes = {_scene(by_id[sid]) for sid in train}
    validation_scenes = {_scene(by_id[sid]) for sid in validation}
    train_groups = {by_id[sid]["group_id"] for sid in train}
    validation_groups = {by_id[sid]["group_id"] for sid in validation}
    if not train or not validation or train_scenes & validation_scenes or train_groups & validation_groups:
        raise ValueError("training/validation scene or group overlap")
    return {"protocol": "geoprr_v2_syncg_scene_disjoint_v1", "seed": seed,
            "scene_disjoint": True, "train_sample_ids": sorted(train),
            "validation_sample_ids": sorted(validation), "train_scene_stems": sorted(train_scenes),
            "validation_scene_stems": sorted(validation_scenes)}


def prepare(root: Path, output_dir: Path, *, seed: int = 20261019,
            validation_fraction: float = 0.1, existing_split: Path | None = None,
            diagnostic_limit: int | None = None) -> dict[str, Any]:
    root = root.resolve()
    if not (root / "annotations").is_dir() or not (root / "images").is_dir():
        raise FileNotFoundError("--root must contain annotations/<split> and images/<split>")
    all_rows = {}
    for partition, expected_count in (("train", SYNCG_TRAIN_ROWS), ("test", SYNCG_TEST_ROWS)):
        paths = sorted((root / "annotations" / partition).glob("*.json"))
        if diagnostic_limit is not None:
            if partition == "test":
                continue
            paths = paths[:diagnostic_limit]
        else:
            if len(paths) != expected_count:
                raise ValueError(f"SyncG {partition} upload incomplete: {len(paths)}/{expected_count} annotations; use --diagnostic-limit only for smoke checks")
            if sample_ids_sha256(path.stem for path in paths) != SYNCG_SAMPLE_IDS_SHA256[partition]:
                raise ValueError(f"SyncG {partition} annotation roster differs from the pinned official release")
        if not paths:
            raise ValueError(f"no SyncG {partition} annotations available")
        all_rows[partition] = [syncg_row(path, root / "images" / partition) for path in paths]
        if diagnostic_limit is not None:
            for row in all_rows[partition]:
                row["diagnostic_only"] = True
    split = make_split(all_rows["train"], seed=seed, validation_fraction=validation_fraction, existing_split=existing_split)
    train_ids = set(split["train_sample_ids"])
    diagnostic = diagnostic_limit is not None
    prefix = "diagnostic_" if diagnostic else ""
    outputs = {f"{prefix}source_train.jsonl": all_rows["train"],
               f"{prefix}train.jsonl": [row for row in all_rows["train"] if row["sample_id"] in train_ids],
               f"{prefix}validation.jsonl": [row for row in all_rows["train"] if row["sample_id"] not in train_ids]}
    if not diagnostic:
        outputs["test.jsonl"] = all_rows["test"]
    summary = {"protocol": split["protocol"], "diagnostic_only": diagnostic,
               "root": str(root), "seed": seed, "split": f"{prefix}split.json",
               "counts": {name: len(rows) for name, rows in outputs.items()},
               "train_validation_scene_overlap": 0, "train_validation_group_overlap": 0,
               "official_test_used_for_training_or_selection": False}
    # Resolve every row and the complete split before writing the final roster.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        (output_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output_dir / f"{prefix}split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"{prefix}preparation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split", type=Path, help="reuse an existing scene-disjoint sample-ID split")
    parser.add_argument("--diagnostic-limit", type=int, help="incomplete-upload smoke only; filenames are marked diagnostic")
    args = parser.parse_args()
    if args.diagnostic_limit is not None and args.diagnostic_limit < 2:
        parser.error("--diagnostic-limit must be at least 2")
    print(json.dumps(prepare(args.root, args.output_dir, seed=args.seed,
                             validation_fraction=args.validation_fraction, existing_split=args.split,
                             diagnostic_limit=args.diagnostic_limit), indent=2))


if __name__ == "__main__":
    main()
