"""Matched batch-one efficiency benchmark for DB-GAR18, ResNet18 and SARN-v2.

The benchmark reads a label-free canonical-ROI manifest, constructs one frozen
condition outside the timed region, then times the complete inference path.  In
the SARN-v2 arms the timed path includes support detection, crop/resize, model
preprocessing, host-to-device transfer, forward pass, output validation, and
CUDA synchronization.  PNG I/O and degradation synthesis are excluded.

Run one arm per process so CUDA peak-memory accounting is attributable to that
arm.  The intended paper matrix is two model families x SARN off/on x clean and
perspective-severe, all using the same sample roster and seed-20262020 terminal
checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from experiments import robustness_degradations
from experiments.benchmark_cagh_v5_paper_efficiency import (
    BenchmarkInput,
    BenchmarkTarget,
    EfficiencyBenchmarkError,
    load_benchmark_inputs,
    measure_target,
    parameter_inventory,
)
from experiments.run_cagh_v5_plain_paper_batch import (
    ROBUSTNESS_SEED,
    canonical_roi_pixel_sha256,
)
from experiments.support_aware_roi_normalization_v2 import (
    ALGORITHM as SARN_ALGORITHM,
    PROTOCOL as SARN_PROTOCOL,
    normalize_support_aware_roi_v2,
)


PROTOCOL: Final[str] = "db_gar18_sarn_v2_batch1_efficiency_v1"
FAMILIES: Final[tuple[str, ...]] = ("db_gar18", "db_resnet18")
CONDITIONS: Final[tuple[str, ...]] = ("clean", "perspective_severe")
DEFAULT_MANIFEST: Final[Path] = Path(
    "artifacts/manifests/unified_real_photo_progress_v1/input_manifest.jsonl"
)
DEFAULT_LIMIT: Final[int] = 100
DEFAULT_WARMUP: Final[int] = 20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EfficiencyBenchmarkError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prepare_condition_inputs(
    inputs: Sequence[BenchmarkInput],
    *,
    condition: str,
) -> tuple[tuple[BenchmarkInput, ...], dict[str, Any]]:
    """Materialize the frozen condition outside the timing window."""

    _require(condition in CONDITIONS, f"unsupported benchmark condition: {condition}")
    _require(bool(inputs), "benchmark input roster is empty")
    prepared: list[BenchmarkInput] = []
    hashes: list[dict[str, str]] = []
    for item in inputs:
        image, metadata = robustness_degradations.apply_degradation(
            item.image_bgr,
            condition,
            sample_id=item.sample_id,
            seed=ROBUSTNESS_SEED,
        )
        _require(
            metadata.get("condition") == condition,
            "degradation metadata condition mismatch",
        )
        image = np.ascontiguousarray(image)
        prepared.append(BenchmarkInput(sample_id=item.sample_id, image_bgr=image))
        hashes.append(
            {
                "sample_id": item.sample_id,
                "condition_pixel_sha256": canonical_roi_pixel_sha256(image),
            }
        )
    return tuple(prepared), {
        "condition": condition,
        "robustness_seed": ROBUSTNESS_SEED,
        "degradation_protocol": robustness_degradations.ROBUSTNESS_PROTOCOL,
        "condition_roster_sha256": _canonical_sha256(hashes),
        "degradation_synthesis_in_timed_region": False,
    }


def build_target(
    family: str,
    *,
    checkpoint_path: Path,
    device: str,
    with_sarn: bool,
) -> BenchmarkTarget:
    """Load one frozen matched checkpoint and optionally prepend SARN-v2."""

    _require(family in FAMILIES, f"unknown model family: {family}")
    checkpoint = Path(checkpoint_path).resolve()
    _require(checkpoint.is_file(), f"checkpoint does not exist: {checkpoint}")
    if family == "db_gar18":
        from experiments.domain_balanced_geoattn_resnet18 import (
            load_checkpoint_predictor,
        )
    else:
        from experiments.domain_balanced_resnet18 import load_checkpoint_predictor

    loaded_method, batch_predict = load_checkpoint_predictor(
        checkpoint,
        device_name=device,
    )

    def predict(image_bgr: np.ndarray) -> float:
        model_input = np.ascontiguousarray(image_bgr)
        if with_sarn:
            model_input = normalize_support_aware_roi_v2(model_input).image
        values = batch_predict((model_input,))
        _require(len(values) == 1, "predictor returned the wrong batch size")
        return float(values[0])

    suffix = "+SARN-v2" if with_sarn else ""
    return BenchmarkTarget(
        method=f"{family}{'_sarn_v2' if with_sarn else '_base'}",
        display_name=f"{loaded_method}{suffix}",
        predict=predict,
        parameter_roots=(batch_predict,),
    )


def audit_sarn_application(inputs: Sequence[BenchmarkInput]) -> dict[str, Any]:
    """Count deterministic SARN decisions outside the timing window."""

    applied = 0
    fallbacks: Counter[str] = Counter()
    for item in inputs:
        result = normalize_support_aware_roi_v2(item.image_bgr)
        if result.applied:
            applied += 1
        else:
            fallbacks[str(result.fallback_reason or "no_op")] += 1
    return {
        "algorithm": SARN_ALGORITHM,
        "protocol": SARN_PROTOCOL,
        "learned_parameters": 0,
        "rows": len(inputs),
        "applied_rows": applied,
        "fallback_or_no_op_rows": len(inputs) - applied,
        "fallback_or_no_op_reasons": dict(sorted(fallbacks.items())),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    measurement = report["measurement"]
    latency = measurement["latency"]
    memory = measurement["cuda_memory"]
    peak = (
        f"{float(memory['peak_allocated_mib']):.1f}"
        if memory.get("supported")
        else "N/A"
    )
    front_end = report["front_end"]
    return "\n".join(
        [
            "# DB-GAR18 / matched ResNet18 efficiency",
            "",
            "PNG decoding and degradation generation are excluded; SARN-v2 is included when enabled.",
            "",
            "| Model arm | Condition | Params (M) | Mean (ms) | P50 (ms) | P95 (ms) | Throughput (img/s) | Peak CUDA MiB | SARN applied |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {report['display_name']} | {report['condition']} | "
                f"{int(report['parameters']['total_parameters']) / 1_000_000.0:.3f} | "
                f"{float(latency['mean_ms']):.3f} | {float(latency['p50_ms']):.3f} | "
                f"{float(latency['p95_ms']):.3f} | "
                f"{float(latency['throughput_images_per_second']):.3f} | {peak} | "
                f"{int(front_end['applied_rows'])}/{int(front_end['rows'])} |"
            ),
            "",
            f"Roster: {measurement['timed_calls']} images, warmup {measurement['warmup_calls']}, batch size 1.",
            "",
        ]
    )


def run_benchmark(
    *,
    family: str,
    checkpoint_path: Path,
    manifest_path: Path,
    condition: str,
    with_sarn: bool,
    output_json: Path,
    output_markdown: Path | None,
    device_name: str,
    limit: int,
    warmup: int,
    target_factory: Callable[..., BenchmarkTarget] = build_target,
) -> dict[str, Any]:
    clean_inputs, input_identity = load_benchmark_inputs(manifest_path, limit=limit)
    inputs, condition_identity = prepare_condition_inputs(
        clean_inputs,
        condition=condition,
    )
    target = target_factory(
        family,
        checkpoint_path=Path(checkpoint_path),
        device=device_name,
        with_sarn=with_sarn,
    )
    expected_method = f"{family}{'_sarn_v2' if with_sarn else '_base'}"
    _require(target.method == expected_method, "target factory returned the wrong arm")
    parameters = parameter_inventory(target.parameter_roots)
    measurement = measure_target(
        target,
        inputs,
        device_name=device_name,
        warmup=warmup,
    )
    # Keep the decision audit outside and *after* timing.  Running it before
    # the benchmark would give SARN arms an extra full-roster CPU warmup that
    # the base arms do not receive; both arms should get exactly ``warmup``
    # untimed calls before their measured calls.
    front_end = (
        audit_sarn_application(inputs)
        if with_sarn
        else {
            "algorithm": "none",
            "protocol": None,
            "learned_parameters": 0,
            "rows": len(inputs),
            "applied_rows": 0,
            "fallback_or_no_op_rows": len(inputs),
            "fallback_or_no_op_reasons": {"disabled": len(inputs)},
        }
    )
    front_end["decision_audit_after_timing"] = True
    measurement["timing_scope"]["included"].insert(
        0,
        "SARN-v2 support detection, crop, and resize" if with_sarn else "no learned or ROI front-end",
    )
    measurement["timing_scope"]["excluded"].append("robustness degradation synthesis")
    device = torch.device(device_name)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "method": target.method,
        "display_name": target.display_name,
        "condition": condition,
        "with_sarn_v2": bool(with_sarn),
        "checkpoint": {
            "path": str(Path(checkpoint_path).resolve()),
            "sha256": _sha256_file(Path(checkpoint_path)),
        },
        "input": input_identity,
        "condition_input": condition_identity,
        "front_end": front_end,
        "environment": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        },
        "parameters": parameters,
        "measurement": measurement,
    }
    output = Path(output_json).resolve()
    _require(output != Path(manifest_path).resolve(), "output cannot overwrite manifest")
    _require(output != Path(checkpoint_path).resolve(), "output cannot overwrite checkpoint")
    _write_json(output, report)
    markdown = output.with_suffix(".md") if output_markdown is None else Path(output_markdown)
    markdown = markdown.resolve()
    _require(markdown != output, "JSON and Markdown outputs must differ")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--with-sarn", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(
        family=args.family,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        condition=args.condition,
        with_sarn=args.with_sarn,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        device_name=args.device,
        limit=args.limit,
        warmup=args.warmup,
    )
    print(json.dumps({"status": report["status"], "method": report["method"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "FAMILIES",
    "PROTOCOL",
    "audit_sarn_application",
    "build_target",
    "prepare_condition_inputs",
    "run_benchmark",
]
