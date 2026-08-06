"""Audit OCR-corpus exposure against every frozen GARC public partition.

The audit is metadata-only.  It authenticates the existing OCR corpus and the
four label-free GARC rosters, then reports exact sample/group intersections for
each OCR inner partition.  If supplied, the frozen 412/19 joint-OOF roster is
audited as well.  No image or annotation file is opened by this module.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range_public_protocol import (
    PARTITIONS as GARC_PARTITIONS,
    canonical_bytes,
    canonical_sha256,
    guard_public_path,
    load_frozen_protocol,
    load_partition_roster,
    require,
    sha256_file,
    strict_json,
    strict_jsonl,
)
from experiments.garc_full_auto_public import load_joint_oof_cohort
from experiments.syncg_numeric_ocr import PROTOCOL as OCR_PROTOCOL


AUDIT_PROTOCOL: Final[str] = "syncg_ocr_garc_partition_overlap_audit_v1"
OCR_PARTITIONS: Final[tuple[str, ...]] = ("train", "calibration", "validation")
OUTER_PARTITIONS: Final[tuple[str, ...]] = (
    "calibration",
    "development_excluded",
    "independent_validation",
)
DEFAULT_OCR_CORPUS: Final[Path] = Path(r"C:\pointer_read\syncg_numeric_ocr_public_v1")
DEFAULT_GARC_PROTOCOL: Final[Path] = Path(
    r"C:\pointer_read\automatic_numeric_range_public_protocol_20260806_v1\protocol.json"
)
DEFAULT_JOINT_OOF: Final[Path] = Path(
    r"C:\pointer_read\garc_full_auto_public_v1\manifests\summary.json"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    r"C:\pointer_read\syncg_ocr_garc_overlap_audit_v1\audit.json"
)


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    require(not output.exists(), f"refusing to overwrite audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(dict(value), pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _load_ocr_corpus(root: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    corpus_root = Path(root).resolve(strict=True)
    require(corpus_root.is_dir(), "OCR corpus root is not a directory")
    summary_path = corpus_root / "summary.json"
    samples_path = corpus_root / "samples.jsonl"
    tokens_path = corpus_root / "tokens.jsonl"
    seal_path = corpus_root / "seal.json"
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("protocol") == OCR_PROTOCOL, "OCR corpus protocol drift")
    require(summary.get("status") == "complete", "OCR corpus is incomplete")
    require(seal.get("protocol") == OCR_PROTOCOL, "OCR seal protocol drift")
    require(seal.get("status") == "sealed", "OCR corpus is not sealed")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "OCR summary seal drift")
    require(
        summary.get("artifacts", {}).get("samples_sha256")
        == seal.get("samples_sha256")
        == sha256_file(samples_path),
        "OCR sample artifact drift",
    )
    require(
        summary.get("artifacts", {}).get("tokens_sha256")
        == seal.get("tokens_sha256")
        == sha256_file(tokens_path),
        "OCR token artifact drift",
    )
    rows = strict_jsonl(samples_path)
    require(len(rows) == int(summary["inventory"]["samples"]), "OCR sample count drift")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        partition = str(row.get("partition") or "")
        require(bool(sample_id and group_id), f"OCR sample[{index}] lacks identity")
        require(sample_id not in seen, f"duplicate OCR sample: {sample_id}")
        require(partition in OCR_PARTITIONS, f"unknown OCR partition: {partition}")
        seen.add(sample_id)
    return corpus_root, summary, rows


def identity_sets(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    return (
        {str(row["sample_id"]) for row in rows},
        {str(row["group_id"]) for row in rows},
    )


def overlap_record(
    left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left_samples, left_groups = identity_sets(left_rows)
    right_samples, right_groups = identity_sets(right_rows)
    sample_overlap = sorted(left_samples & right_samples)
    group_overlap = sorted(left_groups & right_groups)
    return {
        "left_samples": len(left_samples),
        "left_groups": len(left_groups),
        "right_samples": len(right_samples),
        "right_groups": len(right_groups),
        "sample_overlap": len(sample_overlap),
        "group_overlap": len(group_overlap),
        "sample_overlap_ids_sha256": canonical_sha256(sample_overlap),
        "group_overlap_ids_sha256": canonical_sha256(group_overlap),
    }


def build_overlap_matrix(
    ocr_rows: Sequence[Mapping[str, Any]],
    garc_rosters: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_ocr: dict[str, list[Mapping[str, Any]]] = {
        partition: [row for row in ocr_rows if row.get("partition") == partition]
        for partition in OCR_PARTITIONS
    }
    by_ocr["all"] = list(ocr_rows)
    return {
        ocr_partition: {
            garc_partition: overlap_record(rows, garc_rows)
            for garc_partition, garc_rows in garc_rosters.items()
        }
        for ocr_partition, rows in by_ocr.items()
    }


def audit(
    *,
    ocr_corpus: Path,
    garc_protocol: Path,
    output: Path,
    joint_oof_summary: Path | None = None,
) -> Path:
    corpus_root, corpus_summary, ocr_rows = _load_ocr_corpus(ocr_corpus)
    protocol_path, protocol = load_frozen_protocol(garc_protocol)
    garc_rosters: dict[str, list[dict[str, Any]]] = {}
    roster_sources: dict[str, Any] = {}
    for partition in GARC_PARTITIONS:
        _, manifest, rows, roster_audit = load_partition_roster(protocol_path, partition)
        garc_rosters[partition] = rows
        roster_sources[partition] = {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "samples": roster_audit["samples"],
            "groups": roster_audit["groups"],
            "sample_ids_sha256": roster_audit["sample_ids_sha256"],
            "group_ids_sha256": roster_audit["group_ids_sha256"],
        }

    matrix = build_overlap_matrix(ocr_rows, garc_rosters)
    joint_record: dict[str, Any] | None = None
    joint_source: dict[str, Any] | None = None
    if joint_oof_summary is not None:
        joint_path = guard_public_path(joint_oof_summary, label="joint OOF summary")
        joint_summary, joint_rows, _ = load_joint_oof_cohort(joint_path)
        joint_record = {
            partition: overlap_record(
                [row for row in ocr_rows if row.get("partition") == partition],
                joint_rows,
            )
            for partition in (*OCR_PARTITIONS,)
        }
        joint_record["all"] = overlap_record(ocr_rows, joint_rows)
        joint_source = {
            "path": str(joint_path),
            "sha256": sha256_file(joint_path),
            "samples": int(joint_summary["joint_oof"]["samples"]),
            "groups": int(joint_summary["joint_oof"]["groups"]),
            "sample_ids_sha256": joint_summary["joint_oof"]["sample_ids_sha256"],
            "group_ids_sha256": joint_summary["joint_oof"]["group_ids_sha256"],
        }

    train_outer_overlap = {
        partition: matrix["train"][partition] for partition in OUTER_PARTITIONS
    }
    contaminated = any(value["group_overlap"] > 0 for value in train_outer_overlap.values())
    if joint_record is not None:
        contaminated = contaminated or joint_record["train"]["group_overlap"] > 0
    result = {
        "schema_version": 1,
        "protocol": AUDIT_PROTOCOL,
        "status": "overlap_detected_training_evidence_ineligible" if contaminated else "passed",
        "ocr_corpus": {
            "root": str(corpus_root),
            "summary_sha256": sha256_file(corpus_root / "summary.json"),
            "seal_sha256": sha256_file(corpus_root / "seal.json"),
            "samples_sha256": corpus_summary["artifacts"]["samples_sha256"],
            "tokens_sha256": corpus_summary["artifacts"]["tokens_sha256"],
            "samples": int(corpus_summary["inventory"]["samples"]),
            "groups": int(corpus_summary["inventory"]["groups"]),
            "inner_train_groups": int(corpus_summary["split"]["partitions"]["train"]["groups"]),
        },
        "garc_parent_protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "identity": protocol["protocol"],
        },
        "garc_label_free_rosters": roster_sources,
        "overlap_matrix": matrix,
        "joint_oof_412_19": {
            "source": joint_source,
            "overlap_by_ocr_partition": joint_record,
        },
        "eligibility": {
            "existing_tiny_or_strong_trained_on_this_corpus_all_component_oof_eligible": False
            if contaminated
            else True,
            "reason": (
                "OCR training groups intersect frozen outer GARC groups; OCR is not unseen on the claimed cohort"
                if contaminated
                else "OCR training groups are disjoint from all frozen outer GARC groups"
            ),
            "required_remediation": (
                "rebuild OCR corpus from algorithm_fit groups only and retrain detector/recognizers"
                if contaminated
                else None
            ),
        },
        "audit": {
            "public_images_opened": 0,
            "public_annotations_opened": 0,
            "outer_numeric_values_opened": 0,
            "restricted_namespace_images_opened": 0,
            "metadata_only": True,
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    return _atomic_new_json(output, result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-corpus", type=Path, default=DEFAULT_OCR_CORPUS)
    parser.add_argument("--garc-protocol", type=Path, default=DEFAULT_GARC_PROTOCOL)
    parser.add_argument("--joint-oof-summary", type=Path, default=DEFAULT_JOINT_OOF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    path = audit(
        ocr_corpus=args.ocr_corpus,
        garc_protocol=args.garc_protocol,
        joint_oof_summary=args.joint_oof_summary,
        output=args.output,
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_PROTOCOL",
    "build_overlap_matrix",
    "identity_sets",
    "overlap_record",
    "audit",
]
