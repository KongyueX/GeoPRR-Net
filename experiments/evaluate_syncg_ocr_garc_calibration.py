"""Evaluate Tiny OCR recognition on the frozen outer GARC calibration groups.

This is a *component* evaluation: public Text boxes are used only to isolate
recognizer crops.  It is therefore suitable for the pre-frozen Tiny-to-Strong
recognizer upgrade gate, but it is not an end-to-end range-reading result.

The formal path admits exactly the 2,224 images / 100 physical groups in the
frozen GARC ``calibration`` roster.  ``independent_validation``,
``development_excluded``, joint-OOF, field, test, sealed, and confirmatory
images/annotations are never admitted.  ``--validate-only`` authenticates
metadata and model/corpus identities without opening any image or annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range_public_protocol import (
    canonical_bytes,
    canonical_sha256,
    load_frozen_protocol,
    load_partition_roster,
    require,
    sha256_file,
    strict_json,
    strict_jsonl,
    verify_bound_file,
)
from experiments.build_garc_aligned_syncg_numeric_ocr_public import (
    ALIGNMENT_PROTOCOL,
    PUBLIC_ANNOTATION_ROOT,
    PUBLIC_IMAGE_ROOT,
)
from experiments.syncg_numeric_ocr import (
    CHECKPOINT_PROTOCOL,
    PROTOCOL as OCR_PROTOCOL,
    VOCABULARY,
    OCRCorpus,
    SyncGTextRecognitionImageDataset,
    TinyCTCRecognizer,
    canonical_roi_bounds,
    normalize_bbox_to_roi,
    normalize_numeric_text,
)
from experiments.train_syncg_numeric_ocr import evaluate_recognizer
from experiments.syncg_strong_numeric_ocr import (
    STRONG_CHECKPOINT_PROTOCOL,
    MobileSVTRCTCRecognizer,
)


EVALUATION_PROTOCOL: Final[str] = "syncg_garc_calibration_recognizer_component_v1"
SAFE_OUTPUT_ROOT: Final[Path] = Path(r"C:\pointer_read").resolve()
DEFAULT_CORPUS: Final[Path] = SAFE_OUTPUT_ROOT / "syncg_numeric_ocr_garc_aligned_v1"
DEFAULT_TINY_SUMMARY: Final[Path] = (
    SAFE_OUTPUT_ROOT / "syncg_numeric_ocr_garc_aligned_runs/seed_20260817/summary.json"
)
DEFAULT_OUTPUT: Final[Path] = (
    SAFE_OUTPUT_ROOT
    / "syncg_garc_calibration_tiny_ocr_seed_20260817_v1/summary.json"
)
EXPECTED_SEED: Final[int] = 20260817
DEFAULT_STRONG_SUMMARY: Final[Path] = (
    SAFE_OUTPUT_ROOT
    / "syncg_strong_numeric_ocr_garc_aligned_runs/seed_20260818/summary.json"
)
DEFAULT_STRONG_OUTPUT: Final[Path] = (
    SAFE_OUTPUT_ROOT
    / "syncg_garc_calibration_strong_ocr_seed_20260818_v1/summary.json"
)
EXPECTED_STRONG_SEED: Final[int] = 20260818
EXPECTED_CALIBRATION: Final[tuple[int, int]] = (2_224, 100)
FORBIDDEN_EVALUATION_PARTITIONS: Final[tuple[str, ...]] = (
    "development_excluded",
    "independent_validation",
    "joint_oof_412_19",
)


def _safe_existing(path: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay below {SAFE_OUTPUT_ROOT}") from error
    return resolved


def _safe_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"calibration report must stay below {SAFE_OUTPUT_ROOT}") from error
    require(resolved != SAFE_OUTPUT_ROOT, "refusing broad calibration output root")
    return resolved


def _under_public(path: Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(Path(root).resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes public SyncG/train: {resolved}") from error
    return resolved


def _corpus_identity(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = _safe_existing(root, label="aligned OCR corpus")
    summary_path = resolved / "summary.json"
    seal_path = resolved / "seal.json"
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("protocol") == OCR_PROTOCOL, "aligned corpus protocol drift")
    require(summary.get("alignment_protocol") == ALIGNMENT_PROTOCOL, "alignment protocol drift")
    require(summary.get("status") == "complete", "aligned corpus incomplete")
    require(seal.get("status") == "sealed", "aligned corpus is not sealed")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "corpus summary seal drift")
    require(
        seal.get("samples_sha256") == summary.get("artifacts", {}).get("samples_sha256"),
        "corpus sample binding drift",
    )
    require(
        seal.get("tokens_sha256") == summary.get("artifacts", {}).get("tokens_sha256"),
        "corpus token binding drift",
    )
    require(
        summary.get("alignment_audit", {}).get("all_outer_group_overlap_zero") is True
        and summary.get("alignment_audit", {}).get("all_outer_sample_overlap_zero") is True,
        "aligned corpus does not prove outer exclusion",
    )
    identity = {
        "root": str(resolved),
        "summary_sha256": sha256_file(summary_path),
        "seal_sha256": sha256_file(seal_path),
        "samples_sha256": summary["artifacts"]["samples_sha256"],
        "tokens_sha256": summary["artifacts"]["tokens_sha256"],
        "samples": int(summary["inventory"]["samples"]),
        "groups": int(summary["inventory"]["groups"]),
        "alignment_protocol": ALIGNMENT_PROTOCOL,
    }
    require((identity["samples"], identity["groups"]) == (12_176, 551), "fit corpus inventory drift")
    return resolved, summary, identity


def _checkpoint_inputs(
    tiny_summary_path: Path, *, corpus_identity: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], Path, Mapping[str, Any]]:
    summary_path = _safe_existing(tiny_summary_path, label="Tiny summary")
    summary = strict_json(summary_path)
    require(summary.get("protocol") == OCR_PROTOCOL, "Tiny summary protocol drift")
    require(summary.get("status") == "complete", "Tiny summary incomplete")
    require(summary.get("mode") == "formal", "Tiny summary is not formal")
    require(int(summary.get("seed", -1)) == EXPECTED_SEED, "Tiny seed drift")
    require(summary.get("component_selection") == "both", "Tiny run lacks both components")
    expected_checkpoint_corpus = {
        key: corpus_identity[key]
        for key in ("root", "summary_sha256", "samples_sha256", "tokens_sha256")
    }
    require(summary.get("corpus") == expected_checkpoint_corpus, "Tiny summary corpus drift")
    checkpoint_path = _safe_existing(
        Path(str(summary.get("artifacts", {}).get("recognizer") or "")),
        label="Tiny recognizer",
    )
    require(
        summary.get("artifact_sha256", {}).get("recognizer") == sha256_file(checkpoint_path),
        "Tiny recognizer summary hash drift",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping), "Tiny checkpoint is not a mapping")
    require(checkpoint.get("protocol") == CHECKPOINT_PROTOCOL, "Tiny checkpoint protocol drift")
    require(checkpoint.get("status") == "complete", "Tiny checkpoint incomplete")
    require(checkpoint.get("mode") == "formal", "Tiny checkpoint is not formal")
    require(checkpoint.get("component") == "recognizer", "Tiny checkpoint role drift")
    require(int(checkpoint.get("seed", -1)) == EXPECTED_SEED, "Tiny checkpoint seed drift")
    require(checkpoint.get("vocabulary") == list(VOCABULARY), "Tiny vocabulary drift")
    require(checkpoint.get("corpus") == expected_checkpoint_corpus, "Tiny checkpoint corpus drift")
    require(isinstance(checkpoint.get("state_dict"), Mapping), "Tiny checkpoint state missing")
    return summary_path, summary, checkpoint_path, checkpoint


def _strong_checkpoint_inputs(
    strong_summary_path: Path, *, corpus_identity: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], Path, Mapping[str, Any]]:
    summary_path = _safe_existing(strong_summary_path, label="Strong summary")
    summary = strict_json(summary_path)
    require(summary.get("protocol") == STRONG_CHECKPOINT_PROTOCOL, "Strong summary protocol drift")
    require(summary.get("status") == "complete", "Strong summary incomplete")
    require(summary.get("mode") == "formal", "Strong summary is not formal")
    require(int(summary.get("seed", -1)) == EXPECTED_STRONG_SEED, "Strong seed drift")
    expected_checkpoint_corpus = {
        key: corpus_identity[key]
        for key in ("root", "summary_sha256", "samples_sha256", "tokens_sha256")
    }
    require(summary.get("corpus") == expected_checkpoint_corpus, "Strong summary corpus drift")
    checkpoint_path = _safe_existing(Path(str(summary.get("artifact") or "")), label="Strong recognizer")
    require(summary.get("artifact_sha256") == sha256_file(checkpoint_path), "Strong recognizer summary hash drift")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping), "Strong checkpoint is not a mapping")
    require(checkpoint.get("protocol") == STRONG_CHECKPOINT_PROTOCOL, "Strong checkpoint protocol drift")
    require(checkpoint.get("status") == "complete", "Strong checkpoint incomplete")
    require(checkpoint.get("mode") == "formal", "Strong checkpoint is not formal")
    require(checkpoint.get("component") == "recognizer", "Strong checkpoint role drift")
    require(int(checkpoint.get("seed", -1)) == EXPECTED_STRONG_SEED, "Strong checkpoint seed drift")
    require(checkpoint.get("vocabulary") == list(VOCABULARY), "Strong vocabulary drift")
    require(checkpoint.get("corpus") == expected_checkpoint_corpus, "Strong checkpoint corpus drift")
    require(isinstance(checkpoint.get("state_dict"), Mapping), "Strong checkpoint state missing")
    return summary_path, summary, checkpoint_path, checkpoint


def authenticate_metadata(
    *, corpus_root: Path, tiny_summary_path: Path, recognizer_kind: str = "tiny"
) -> dict[str, Any]:
    """Authenticate the exact fit corpus, Tiny model, and outer calibration roster."""

    corpus_path, corpus_summary, corpus_identity = _corpus_identity(corpus_root)
    require(recognizer_kind in ("tiny", "strong"), "unknown recognizer kind")
    if recognizer_kind == "tiny":
        recognizer_summary_file, recognizer_summary, checkpoint_file, _ = _checkpoint_inputs(
            tiny_summary_path, corpus_identity=corpus_identity
        )
    else:
        recognizer_summary_file, recognizer_summary, checkpoint_file, _ = _strong_checkpoint_inputs(
            tiny_summary_path, corpus_identity=corpus_identity
        )
    parent_path, parent = load_frozen_protocol(
        Path(str(corpus_summary["parent_garc_protocol"]["path"]))
    )
    require(
        sha256_file(parent_path) == corpus_summary["parent_garc_protocol"]["sha256"],
        "parent GARC protocol drift",
    )
    _, roster_path, roster, roster_audit = load_partition_roster(
        parent_path, "calibration"
    )
    binding = corpus_summary["garc_partition_bindings"]["calibration"]
    require(str(roster_path) == binding["path"], "calibration roster path drift")
    require(sha256_file(roster_path) == binding["sha256"], "calibration roster hash drift")
    for key in ("samples", "groups", "sample_ids_sha256", "group_ids_sha256"):
        require(roster_audit[key] == binding[key], f"calibration {key} drift")
    require(
        (int(roster_audit["samples"]), int(roster_audit["groups"]))
        == EXPECTED_CALIBRATION,
        "calibration inventory drift",
    )
    fit = corpus_summary["alignment_audit"]["algorithm_fit_exact_coverage"]
    outer = corpus_summary["alignment_audit"]["outer_exclusion"]["calibration"]
    require(int(outer["sample_overlap"]) == 0, "fit/calibration sample overlap")
    require(int(outer["group_overlap"]) == 0, "fit/calibration group overlap")
    manifest_path = verify_bound_file(parent, "source_bindings", "syncg_train_manifest")
    result = {
        "protocol": EVALUATION_PROTOCOL,
        "status": "metadata_authenticated_no_images_or_annotations_opened",
        "partition": "calibration",
        "evaluation_kind": "recognizer_oracle_text_boxes_component_only",
        "recognizer_kind": recognizer_kind,
        "training_corpus": corpus_identity,
        "recognizer_summary": {
            "path": str(recognizer_summary_file),
            "sha256": sha256_file(recognizer_summary_file),
            "seed": int(recognizer_summary["seed"]),
        },
        "recognizer_checkpoint": {
            "path": str(checkpoint_file),
            "sha256": sha256_file(checkpoint_file),
        },
        "parent_garc_protocol": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "identity": parent["protocol"],
        },
        "calibration_roster": {
            "path": str(roster_path),
            "sha256": sha256_file(roster_path),
            **{key: roster_audit[key] for key in (
                "samples", "groups", "sample_ids_sha256", "group_ids_sha256"
            )},
        },
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "disjointness": {
            "training_sample_ids_sha256": fit["sample_ids_sha256"],
            "training_group_ids_sha256": fit["group_ids_sha256"],
            "calibration_sample_ids_sha256": roster_audit["sample_ids_sha256"],
            "calibration_group_ids_sha256": roster_audit["group_ids_sha256"],
            "sample_overlap": 0,
            "group_overlap": 0,
        },
        "data_access_audit": {
            "calibration_images_opened": 0,
            "calibration_annotations_opened": 0,
            "algorithm_fit_images_opened": 0,
            "inner_validation_images_opened": 0,
            "development_excluded_images_opened": 0,
            "development_excluded_annotations_opened": 0,
            "independent_validation_images_opened": 0,
            "independent_validation_annotations_opened": 0,
            "joint_oof_412_19_samples_opened": 0,
            "field_samples_opened": 0,
            "public_test_samples_opened": 0,
            "sealed_samples_opened": 0,
            "confirmatory_samples_opened": 0,
        },
        "gpu_work_started": False,
    }
    role = "tiny" if recognizer_kind == "tiny" else "strong"
    result[f"{role}_summary"] = dict(result["recognizer_summary"])
    result[f"{role}_checkpoint"] = dict(result["recognizer_checkpoint"])
    return result


def _calibration_component_corpus(
    metadata: Mapping[str, Any], *, project_root: Path
) -> tuple[OCRCorpus, dict[str, Any]]:
    """Open only public calibration annotations and build oracle-box token rows."""

    roster = strict_jsonl(Path(metadata["calibration_roster"]["path"]))
    manifest = strict_jsonl(Path(metadata["source_manifest"]["path"]))
    calibration_ids = {str(row["sample_id"]) for row in roster}
    calibration_groups = {str(row["group_id"]) for row in roster}
    require(len(calibration_ids) == EXPECTED_CALIBRATION[0], "calibration ID drift")
    require(len(calibration_groups) == EXPECTED_CALIBRATION[1], "calibration group drift")
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    require(calibration_ids <= set(manifest_by_id), "calibration sample absent from manifest")
    roster_by_id = {str(row["sample_id"]): row for row in roster}

    samples: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    annotation_inventory: list[dict[str, Any]] = []
    for sample_id in sorted(calibration_ids):
        roster_row = roster_by_id[sample_id]
        source = manifest_by_id[sample_id]
        group_id = str(roster_row["group_id"])
        require(str(source["group_id"]) == group_id, f"{sample_id}: group drift")
        metadata_row = source.get("metadata") or {}
        image_path = _under_public(
            Path(str(source.get("image_path") or "")),
            PUBLIC_IMAGE_ROOT,
            label=f"{sample_id}.image",
        )
        annotation_path = _under_public(
            Path(str(metadata_row.get("annotation_path") or "")),
            PUBLIC_ANNOTATION_ROOT,
            label=f"{sample_id}.annotation",
        )
        raw = annotation_path.read_bytes()
        annotation = json.loads(raw.decode("utf-8"))
        require(isinstance(annotation, Mapping), f"{sample_id}: bad annotation")
        require(str(annotation.get("file_name")) == sample_id, f"{sample_id}: annotation drift")
        width, height = int(annotation["width"]), int(annotation["height"])
        dial_bbox = [float(value) for value in annotation["dial_bbox_annotations"][:4]]
        require(
            np.allclose(dial_bbox, [float(v) for v in roster_row["dial_bbox"][:4]], atol=1e-6),
            f"{sample_id}: dial bbox drift",
        )
        roi_bounds = canonical_roi_bounds((height, width, 3), dial_bbox)
        sample_tokens: list[dict[str, Any]] = []
        for index, entry in enumerate(annotation.get("text_bbox_annotations") or []):
            if not isinstance(entry, Mapping) or str(entry.get("type") or "").casefold() != "text":
                continue
            text = normalize_numeric_text(entry.get("value"))
            bbox = [float(value) for value in entry["bbox"][:4]]
            require(len(bbox) == 4 and np.isfinite(bbox).all(), f"{sample_id}: bad text bbox")
            require(bbox[2] > bbox[0] and bbox[3] > bbox[1], f"{sample_id}: empty text bbox")
            token_id = f"{sample_id}:text:{index}"
            normalized = list(normalize_bbox_to_roi(bbox, roi_bounds))
            token = {
                "token_id": token_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "partition": "calibration",
                "image_path": image_path.relative_to(project_root).as_posix(),
                "annotation_path": annotation_path.relative_to(project_root).as_posix(),
                "text": text,
                "numeric_value": float(text),
                "bbox_original": bbox,
                "bbox_roi_normalized": normalized,
            }
            tokens.append(token)
            sample_tokens.append({
                "token_id": token_id,
                "text": text,
                "bbox_roi_normalized": normalized,
            })
        require(bool(sample_tokens), f"{sample_id}: no numeric Text token")
        samples.append({
            "sample_id": sample_id,
            "group_id": group_id,
            "partition": "calibration",
            "image_path": image_path.relative_to(project_root).as_posix(),
            "annotation_path": annotation_path.relative_to(project_root).as_posix(),
            "image_width": width,
            "image_height": height,
            "dial_bbox": dial_bbox,
            "tokens": sample_tokens,
        })
        annotation_inventory.append({
            "sample_id": sample_id,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    require({str(row["sample_id"]) for row in samples} == calibration_ids, "calibration emission drift")
    identity = {
        "samples": len(samples),
        "groups": len({str(row["group_id"]) for row in samples}),
        "tokens": len(tokens),
        "sample_ids_sha256": canonical_sha256(sorted(calibration_ids)),
        "group_ids_sha256": canonical_sha256(sorted(calibration_groups)),
        "token_ids_sha256": canonical_sha256(sorted(str(row["token_id"]) for row in tokens)),
        "annotation_content_inventory_sha256": canonical_sha256(annotation_inventory),
    }
    corpus = OCRCorpus(Path(metadata["training_corpus"]["root"]), tuple(samples), tuple(tokens), {})
    return corpus, identity


def run_formal(
    *, corpus_root: Path, tiny_summary_path: Path, output_path: Path,
    device_name: str, batch_size: int, workers: int, recognizer_kind: str = "tiny",
) -> dict[str, Any]:
    metadata = authenticate_metadata(
        corpus_root=corpus_root, tiny_summary_path=tiny_summary_path,
        recognizer_kind=recognizer_kind,
    )
    loader = _checkpoint_inputs if recognizer_kind == "tiny" else _strong_checkpoint_inputs
    _, _, _, checkpoint = loader(tiny_summary_path, corpus_identity=metadata["training_corpus"])
    component_corpus, component_identity = _calibration_component_corpus(
        metadata, project_root=PROJECT_ROOT.resolve(strict=True)
    )
    dataset = SyncGTextRecognitionImageDataset(
        component_corpus,
        project_root=PROJECT_ROOT,
        partition="calibration",
        seed=EXPECTED_SEED,
        training=False,
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
    model = (
        TinyCTCRecognizer() if recognizer_kind == "tiny" else MobileSVTRCTCRecognizer()
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics = evaluate_recognizer(
        model, dataset, device=device,
        batch_size=max(1, min(int(batch_size), len(dataset))), workers=int(workers),
    )
    require(metrics["source_images"] == EXPECTED_CALIBRATION[0], "evaluation sample drift")
    require(metrics["tokens"] == component_identity["tokens"], "evaluation token drift")
    report = {
        **metadata,
        "status": "formal_calibration_component_evaluation_complete",
        "mode": "formal",
        "component_corpus": component_identity,
        "metrics": metrics,
        "data_access_audit": {
            **metadata["data_access_audit"],
            "calibration_images_opened": EXPECTED_CALIBRATION[0],
            "calibration_annotations_opened": EXPECTED_CALIBRATION[0],
        },
        "gpu_work_started": device.type == "cuda",
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    output = _safe_output(output_path)
    require(not output.exists(), f"refusing to overwrite calibration report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(report, pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--recognizer-kind", choices=("tiny", "strong"), default="tiny")
    parser.add_argument("--recognizer-summary", "--tiny-summary", dest="recognizer_summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = args.recognizer_summary or (
        DEFAULT_TINY_SUMMARY if args.recognizer_kind == "tiny" else DEFAULT_STRONG_SUMMARY
    )
    output = args.output or (
        DEFAULT_OUTPUT if args.recognizer_kind == "tiny" else DEFAULT_STRONG_OUTPUT
    )
    if args.validate_only:
        result = authenticate_metadata(
            corpus_root=args.corpus, tiny_summary_path=summary,
            recognizer_kind=args.recognizer_kind,
        )
    else:
        result = run_formal(
            corpus_root=args.corpus,
            tiny_summary_path=summary,
            output_path=output,
            device_name=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            recognizer_kind=args.recognizer_kind,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_STRONG_OUTPUT",
    "DEFAULT_STRONG_SUMMARY",
    "EVALUATION_PROTOCOL",
    "EXPECTED_CALIBRATION",
    "EXPECTED_SEED",
    "EXPECTED_STRONG_SEED",
    "authenticate_metadata",
    "run_formal",
]
