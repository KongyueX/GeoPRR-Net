"""Fail-closed verifier for GARC-aligned Tiny/Strong OCR evidence.

The verifier authenticates the aligned corpus, proves exact 551-group
``algorithm_fit`` coverage and zero outer overlap, then inspects the completed
Tiny detector/recognizer and Strong recognizer checkpoints.  Every checkpoint
must bind the same corpus summary/samples/tokens hashes.  It opens no image or
annotation and is intended as a mandatory GARC-wrapper preflight.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

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
)
from experiments.build_garc_aligned_syncg_numeric_ocr_public import (
    ALIGNMENT_PROTOCOL,
    assign_inner_partitions,
    verify_corpus,
)
from experiments.syncg_numeric_ocr import CHECKPOINT_PROTOCOL, OCRCorpus
from experiments.syncg_strong_numeric_ocr import STRONG_CHECKPOINT_PROTOCOL


EVIDENCE_PROTOCOL: Final[str] = "garc_aligned_ocr_training_evidence_v1"
DEFAULT_CORPUS: Final[Path] = Path(r"C:\pointer_read\syncg_numeric_ocr_garc_aligned_v1")


def corpus_identity(corpus_root: Path) -> dict[str, Any]:
    corpus = OCRCorpus.load(corpus_root)
    return {
        "root": str(corpus.root),
        "summary_sha256": sha256_file(corpus.root / "summary.json"),
        "samples_sha256": corpus.summary["artifacts"]["samples_sha256"],
        "tokens_sha256": corpus.summary["artifacts"]["tokens_sha256"],
    }


def verify_frozen_inner_assignment(corpus_root: Path) -> dict[str, Any]:
    """Recompute the inner group split from the authenticated GARC fit roster."""

    root = Path(corpus_root).resolve(strict=True)
    summary = strict_json(root / "summary.json")
    parent_path, _ = load_frozen_protocol(Path(summary["parent_garc_protocol"]["path"]))
    _, _, algorithm_fit, _ = load_partition_roster(parent_path, "algorithm_fit")
    local_samples = strict_jsonl(root / "samples.jsonl")
    expected = assign_inner_partitions(
        algorithm_fit,
        seed=int(summary["split"]["seed"]),
        calibration_fraction=float(summary["split"]["calibration_fraction_target"]),
        validation_fraction=float(summary["split"]["validation_fraction_target"]),
    )
    observed = {str(row["sample_id"]): str(row["partition"]) for row in local_samples}
    require(set(observed) == set(expected), "inner split sample inventory drift")
    require(observed == expected, "inner partition assignment differs from frozen hash split")
    return {
        "verified": True,
        "method": summary["split"]["method"],
        "seed": int(summary["split"]["seed"]),
        "assignment_sha256": canonical_sha256(expected),
        "samples": len(expected),
    }


def _load_checkpoint(path: Path, *, label: str) -> tuple[Path, Mapping[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    value = torch.load(resolved, map_location="cpu", weights_only=False)
    require(isinstance(value, Mapping), f"{label} checkpoint is not a mapping")
    return resolved, value


def _checkpoint_binding(
    path: Path,
    *,
    component: str,
    expected_corpus: Mapping[str, Any],
    strong: bool,
) -> dict[str, Any]:
    resolved, value = _load_checkpoint(path, label=component)
    expected_protocol = STRONG_CHECKPOINT_PROTOCOL if strong else CHECKPOINT_PROTOCOL
    require(value.get("protocol") == expected_protocol, f"{component} checkpoint protocol drift")
    require(value.get("status") == "complete", f"{component} checkpoint is not complete")
    require(value.get("mode") == "formal", f"{component} checkpoint is not formal")
    expected_component = "recognizer" if strong else component
    require(value.get("component") == expected_component, f"{component} role drift")
    require(value.get("corpus") == dict(expected_corpus), f"{component} corpus binding drift")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "protocol": expected_protocol,
        "component": value["component"],
        "seed": int(value["seed"]),
        "corpus": dict(value["corpus"]),
    }


def verify_training_evidence(
    *,
    corpus_root: Path,
    tiny_summary_path: Path,
    strong_summary_path: Path | None = None,
) -> dict[str, Any]:
    corpus_verification = verify_corpus(corpus_root)
    require(corpus_verification["protocol"] == ALIGNMENT_PROTOCOL, "wrong aligned corpus")
    require(corpus_verification["samples"] == 12_176, "aligned sample inventory drift")
    require(corpus_verification["groups"] == 551, "aligned group inventory drift")
    require(corpus_verification["algorithm_fit_exact_coverage"] is True, "algorithm_fit coverage absent")
    require(corpus_verification["outer_group_overlap_zero"] is True, "outer group overlap")
    inner_assignment = verify_frozen_inner_assignment(corpus_root)
    expected_corpus = corpus_identity(corpus_root)

    tiny_summary_file = Path(tiny_summary_path).resolve(strict=True)
    tiny = strict_json(tiny_summary_file)
    require(tiny.get("status") == "complete", "Tiny summary is incomplete")
    require(tiny.get("mode") == "formal", "Tiny summary is not formal")
    require(tiny.get("corpus") == expected_corpus, "Tiny summary corpus binding drift")
    require(tiny.get("component_selection") == "both", "Tiny summary lacks detector/recognizer")
    tiny_bindings: dict[str, Any] = {}
    for component in ("detector", "recognizer"):
        artifact = Path(str(tiny.get("artifacts", {}).get(component) or ""))
        binding = _checkpoint_binding(
            artifact,
            component=component,
            expected_corpus=expected_corpus,
            strong=False,
        )
        require(
            binding["sha256"] == tiny.get("artifact_sha256", {}).get(component),
            f"Tiny {component} summary hash drift",
        )
        require(binding["seed"] == int(tiny["seed"]), f"Tiny {component} seed drift")
        tiny_bindings[component] = binding

    strong_evidence: dict[str, Any] = {"available": False}
    if strong_summary_path is not None:
        strong_summary_file = Path(strong_summary_path).resolve(strict=True)
        strong = strict_json(strong_summary_file)
        require(strong.get("protocol") == STRONG_CHECKPOINT_PROTOCOL, "Strong summary protocol drift")
        require(strong.get("status") == "complete", "Strong summary is incomplete")
        require(strong.get("mode") == "formal", "Strong summary is not formal")
        require(strong.get("corpus") == expected_corpus, "Strong summary corpus binding drift")
        strong_artifact = Path(str(strong.get("artifact") or ""))
        strong_binding = _checkpoint_binding(
            strong_artifact,
            component="strong_recognizer",
            expected_corpus=expected_corpus,
            strong=True,
        )
        require(
            strong_binding["sha256"] == strong.get("artifact_sha256"),
            "Strong summary hash drift",
        )
        require(strong_binding["seed"] == int(strong["seed"]), "Strong seed drift")
        strong_evidence = {
            "available": True,
            "summary": {
                "path": str(strong_summary_file),
                "sha256": sha256_file(strong_summary_file),
            },
            "seed": int(strong["seed"]),
            "recognizer": strong_binding,
            "shared_detector_sha256": tiny_bindings["detector"]["sha256"],
        }

    return {
        "schema_version": 1,
        "protocol": EVIDENCE_PROTOCOL,
        "status": "verified_garc_aligned_ocr_training_evidence",
        "corpus": {
            **expected_corpus,
            "seal_sha256": corpus_verification["seal_sha256"],
            "samples": 12_176,
            "groups": 551,
            "algorithm_fit_exact_coverage": True,
            "outer_group_overlap_zero": True,
            "outer_sample_overlap_zero": True,
        },
        "tiny": {
            "summary": {"path": str(tiny_summary_file), "sha256": sha256_file(tiny_summary_file)},
            "seed": int(tiny["seed"]),
            "components": tiny_bindings,
        },
        "strong": strong_evidence,
        "audit": {
            "all_checkpoints_bind_same_corpus": True,
            "frozen_inner_assignment": inner_assignment,
            "public_images_opened": 0,
            "public_annotations_opened": 0,
            "outer_values_opened": 0,
            "restricted_namespace_images_opened": 0,
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    require(not output.exists(), f"refusing to overwrite evidence output: {output}")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tiny-summary", type=Path, required=True)
    parser.add_argument("--strong-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = verify_training_evidence(
        corpus_root=args.corpus,
        tiny_summary_path=args.tiny_summary,
        strong_summary_path=args.strong_summary,
    )
    path = _atomic_new_json(args.output, result)
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "EVIDENCE_PROTOCOL",
    "corpus_identity",
    "verify_frozen_inner_assignment",
    "verify_training_evidence",
]
