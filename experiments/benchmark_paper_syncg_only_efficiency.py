"""Matched batch-one efficiency benchmark for the final SyncG-only models.

Each invocation loads and measures exactly one arm.  Run the five commands as
separate processes so CUDA peak-memory accounting is attributable to one model.
All arms consume the first ``limit`` decoded ROIs from the same label-free
manifest.  PNG decoding and optional degradation synthesis happen before the
timed region; model-native preprocessing, host-to-device transfer, inference,
postprocessing, and CUDA synchronization are timed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.benchmark_cagh_v5_paper_efficiency import (
    BenchmarkTarget,
    EfficiencyBenchmarkError,
    load_benchmark_inputs,
    measure_target,
    parameter_inventory,
)
from experiments.benchmark_db_gar18_sarn_v2_efficiency import (
    CONDITIONS,
    prepare_condition_inputs,
)


PROTOCOL: Final[str] = "paper_syncg_only_batch1_efficiency_v1"
ARMS: Final[tuple[str, ...]] = (
    "direct_resnet18",
    "geoattn_sarn",
    "pgsiam",
    "mobilenet_v3_large",
    "efficientnet_b0",
)
DEFAULT_MANIFEST: Final[Path] = Path(
    "artifacts/manifests/unified_real_photo_progress_v1/input_manifest.jsonl"
)
DEFAULT_LIMIT: Final[int] = 100
DEFAULT_WARMUP: Final[int] = 20


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EfficiencyBenchmarkError(message)


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


def _single_value_predictor(
    batch_predict: Callable[[Sequence[np.ndarray]], list[float]],
    *,
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[[np.ndarray], float]:
    def predict(image_bgr: np.ndarray) -> float:
        model_input = np.ascontiguousarray(image_bgr)
        if preprocess is not None:
            model_input = np.ascontiguousarray(preprocess(model_input))
        values = batch_predict((model_input,))
        _require(len(values) == 1, "predictor returned the wrong batch size")
        return float(values[0])

    return predict


def build_target(
    arm: str,
    *,
    checkpoint_path: Path,
    device: str,
) -> BenchmarkTarget:
    """Load one model through its SyncG-only paper inference loader."""

    _require(arm in ARMS, f"unknown benchmark arm: {arm}")
    checkpoint = Path(checkpoint_path).resolve()
    _require(checkpoint.is_file(), f"checkpoint does not exist: {checkpoint}")

    from experiments.support_aware_roi_normalization_v2 import (
        normalize_support_aware_roi_v2,
    )

    def sarn(image_bgr: np.ndarray) -> np.ndarray:
        return normalize_support_aware_roi_v2(image_bgr).image

    if arm == "direct_resnet18":
        from experiments.resnet18_direct_progress import load_checkpoint_predictor

        loaded_method, batch_predict = load_checkpoint_predictor(
            checkpoint, device_name=device
        )
        _require(
            loaded_method.startswith("resnet18_direct_seed_"),
            "checkpoint is not a Direct-ResNet18 model",
        )
        return BenchmarkTarget(
            method=loaded_method,
            display_name="Direct-ResNet18 + SARN-v2",
            predict=_single_value_predictor(batch_predict, preprocess=sarn),
            parameter_roots=(batch_predict,),
        )

    if arm == "geoattn_sarn":
        from experiments.geoattn_resnet18_progress import load_checkpoint_predictor

        loaded_method, batch_predict = load_checkpoint_predictor(
            checkpoint, device_name=device
        )

        return BenchmarkTarget(
            method=loaded_method,
            display_name="GeoAttn-ResNet18 + SARN-v2",
            predict=_single_value_predictor(batch_predict, preprocess=sarn),
            parameter_roots=(batch_predict,),
        )

    if arm == "pgsiam":
        from experiments.evaluate_projective_geometry_guided_siam import model_batch
        from experiments.projective_geometry_views import (
            build_projective_geometry_views,
        )
        from experiments.train_projective_geometry_guided_siam_syncg import (
            load_checkpoint_model,
        )

        loaded_method, model, _metadata = load_checkpoint_model(
            checkpoint, device_name=device
        )
        model_device = next(model.parameters()).device

        def predict(image_bgr: np.ndarray) -> float:
            views = build_projective_geometry_views(image_bgr)
            rows = model_batch(model, (views,), device=model_device)
            _require(len(rows) == 1, "PG-SIAM returned the wrong batch size")
            return float(rows[0]["trusted_progress"])

        return BenchmarkTarget(
            method=loaded_method,
            display_name="PG-SIAM (SyncG-only)",
            predict=predict,
            parameter_roots=(model,),
        )

    from experiments.syncg_lightweight_regression_baselines import (
        BACKBONE_SPECS,
        load_checkpoint_predictor,
    )

    loaded_method, batch_predict = load_checkpoint_predictor(
        checkpoint, device_name=device
    )
    _require(
        loaded_method.startswith(f"{arm}_seed_"),
        f"checkpoint is not a {arm} model",
    )
    return BenchmarkTarget(
        method=loaded_method,
        display_name=f"{BACKBONE_SPECS[arm].paper_name} + SARN-v2",
        predict=_single_value_predictor(batch_predict, preprocess=sarn),
        parameter_roots=(batch_predict,),
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    measurement = report["measurement"]
    latency = measurement["latency"]
    memory = measurement["cuda_memory"]
    peak = (
        f"{float(memory['peak_allocated_mib']):.1f}"
        if memory.get("supported")
        else "N/A"
    )
    return "\n".join(
        [
            "# SyncG-only batch-1 efficiency",
            "",
            (
                "PNG decoding and degradation synthesis are excluded. Model-native "
                "preprocessing and any SARN/projective front-end are included."
            ),
            "",
            "| Model | Condition | Params (M) | P50 (ms) | P95 (ms) | Throughput (img/s) | Peak CUDA MiB |",
            "|---|---|---:|---:|---:|---:|---:|",
            (
                f"| {report['display_name']} | {report['condition']} | "
                f"{int(report['parameters']['total_parameters']) / 1_000_000.0:.3f} | "
                f"{float(latency['p50_ms']):.3f} | "
                f"{float(latency['p95_ms']):.3f} | "
                f"{float(latency['throughput_images_per_second']):.3f} | {peak} |"
            ),
            "",
            (
                f"Roster: first {measurement['timed_calls']} manifest rows; batch size 1; "
                f"warmup {measurement['warmup_calls']}; one model per process."
            ),
            "",
        ]
    )


def run_benchmark(
    *,
    arm: str,
    checkpoint_path: Path,
    manifest_path: Path,
    condition: str,
    output_json: Path,
    output_markdown: Path | None,
    device_name: str,
    limit: int,
    warmup: int,
    target_factory: Callable[..., BenchmarkTarget] = build_target,
) -> dict[str, Any]:
    clean_inputs, input_identity = load_benchmark_inputs(manifest_path, limit=limit)
    inputs, condition_identity = prepare_condition_inputs(
        clean_inputs, condition=condition
    )
    target = target_factory(
        arm,
        checkpoint_path=Path(checkpoint_path),
        device=device_name,
    )
    parameters = parameter_inventory(target.parameter_roots)
    measurement = measure_target(
        target,
        inputs,
        device_name=device_name,
        warmup=warmup,
    )
    if arm == "pgsiam":
        front_end = "projective dual-view construction, SARN-v2, then PG-SIAM"
    else:
        front_end = "SARN-v2 support detection/crop/resize, then model-native preprocessing"
    measurement["timing_scope"]["included"].insert(0, front_end)
    measurement["timing_scope"]["excluded"].append(
        "robustness degradation synthesis"
    )

    device = torch.device(device_name)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "method": target.method,
        "display_name": target.display_name,
        "condition": condition,
        "checkpoint": {"path": str(Path(checkpoint_path).resolve())},
        "input": input_identity,
        "condition_input": condition_identity,
        "execution": {
            "one_arm_per_invocation": True,
            "independent_process_required": True,
        },
        "environment": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
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
    markdown = (
        output.with_suffix(".md")
        if output_markdown is None
        else Path(output_markdown).resolve()
    )
    _require(markdown != output, "JSON and Markdown outputs must differ")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--condition", choices=CONDITIONS, default="clean")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_benchmark(
        arm=args.arm,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        condition=args.condition,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        device_name=args.device,
        limit=args.limit,
        warmup=args.warmup,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": report["arm"],
                "method": report["method"],
                "output": str(Path(args.output_json).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "DEFAULT_LIMIT",
    "DEFAULT_MANIFEST",
    "DEFAULT_WARMUP",
    "PROTOCOL",
    "build_target",
    "render_markdown",
    "run_benchmark",
]
