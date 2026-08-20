"""Plain label-free batch inference for the eight paper configurations.

This runner is intentionally an ordinary CV program.  It loads exactly one
model per process, applies the six frozen robustness degradations, and writes
normalized progress or an explicit per-image failure.  It has no approval
keys, signatures, authority receipts, one-shot lifecycle, hidden identities,
fault injection, or post-hoc scoring.

The input is either a four-field plain JSONL manifest or the existing
``v5_shared_roi_comparison_input`` consumer JSONL.  Both schemas are strictly
label-free.  Every method receives pixels decoded from the same lossless PNG;
model-native resizing remains inside the corresponding frozen model adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from experiments import robustness_degradations


PROTOCOL: Final[str] = "cagh_v5_plain_paper_batch_v1"
ROBUSTNESS_SEED: Final[int] = 20260720
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
METHODS: Final[tuple[str, ...]] = (
    "full_seed_20262020",
    "full_seed_20262021",
    "full_seed_20262022",
    "control_no_pepd_residual_seed_20262020",
    "control_no_mask_residual_seed_20262020",
    "control_solver_core_seed_20262020",
    "vdn_official200_terminal_seed20",
    "original_transformer_legacy_auto_reference",
)

PLAIN_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {"sample_id", "roi_path", "roi_png_sha256", "roi_pixel_sha256"}
)
SHARED_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "sample_id",
        "group_id",
        "dataset",
        "split",
        "partition",
        "image_path",
        "image_sha256",
        "canonical_roi_sha256",
        "canonical_roi_pixel_sha256",
        "frame_sha256",
        "roi_shape",
        "roi_contract_sha256",
        "materialization_contract_sha256",
        "input_schema_sha256",
    }
)
OUTPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "sample_id",
        "method",
        "condition",
        "robustness_seed",
        "status",
        "normalized_progress",
        "failure_code",
        "roi_png_sha256",
        "roi_pixel_sha256",
        "condition_pixel_sha256",
    }
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_SEALED_ROOT: Final[Path] = Path(
    r"C:\pointer_read\cagh_v5_solver_gated_locked_adaptive_v1"
)


class PlainBatchError(RuntimeError):
    """Invalid batch configuration or input artifact."""


class ModelPredictionFailure(RuntimeError):
    """One model could not produce a valid normalized progress value."""

    def __init__(self, code: str) -> None:
        normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(code)).strip("_")
        super().__init__(normalized[:128] or "model_reported_failure")
        self.code = normalized[:128] or "model_reported_failure"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlainBatchError(message)


def _canonical_json_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_roi_pixel_sha256(image_bgr: np.ndarray) -> str:
    """Match the established canonical BGR pixel-hash domain."""

    image = np.asarray(image_bgr)
    _require(
        image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and min(image.shape[:2]) >= 2,
        "canonical ROI must be uint8 BGR [H,W,3]",
    )
    image = np.ascontiguousarray(image)
    # The established shared-ROI contract hashes canonical JSON bytes without
    # a record-terminating LF, then a NUL separator, then the raw BGR bytes.
    header = _canonical_json_line(
        {
            "protocol": "canonical_meter_roi_bgr_uint8_v1",
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "channel_order": "BGR",
        }
    ).removesuffix(b"\n")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _required_sha256(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} is not lowercase SHA-256",
    )
    return value


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    roi_path: Path
    roi_png_sha256: str
    roi_pixel_sha256: str


def _manifest_row(value: Mapping[str, Any], *, manifest_root: Path) -> ManifestRow:
    keys = set(value)
    _require(
        keys in (set(PLAIN_MANIFEST_KEYS), set(SHARED_MANIFEST_KEYS)),
        "manifest row is not an accepted label-free schema",
    )
    sample_id = value.get("sample_id")
    _require(isinstance(sample_id, str) and bool(sample_id), "sample_id is empty")
    if keys == set(PLAIN_MANIFEST_KEYS):
        path_text = value["roi_path"]
        png_sha256 = _required_sha256(
            value["roi_png_sha256"], label=f"{sample_id}.roi_png"
        )
        pixel_sha256 = _required_sha256(
            value["roi_pixel_sha256"], label=f"{sample_id}.roi_pixels"
        )
    else:
        path_text = value["image_path"]
        png_sha256 = _required_sha256(
            value["canonical_roi_sha256"], label=f"{sample_id}.canonical_roi"
        )
        _require(
            value.get("image_sha256") == png_sha256,
            f"{sample_id}: image and canonical ROI hashes differ",
        )
        pixel_sha256 = _required_sha256(
            value["canonical_roi_pixel_sha256"],
            label=f"{sample_id}.canonical_roi_pixels",
        )
    _require(isinstance(path_text, str) and bool(path_text), f"{sample_id}: ROI path is empty")
    path = Path(path_text)
    if not path.is_absolute():
        path = manifest_root / path
    path = path.resolve()
    _require(path.suffix == ".png", f"{sample_id}: canonical ROI is not lowercase .png")
    return ManifestRow(sample_id, path, png_sha256, pixel_sha256)


def load_manifest(path: Path) -> tuple[ManifestRow, ...]:
    manifest = Path(path).resolve()
    _require(manifest.is_file(), f"manifest does not exist: {manifest}")
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlainBatchError(
                    f"manifest line {line_number} is invalid JSON"
                ) from exc
            _require(isinstance(value, Mapping), f"manifest line {line_number} is not an object")
            row = _manifest_row(value, manifest_root=manifest.parent)
            _require(row.sample_id not in seen, f"duplicate sample_id: {row.sample_id}")
            seen.add(row.sample_id)
            rows.append(row)
    _require(bool(rows), "manifest is empty")
    return tuple(rows)


def load_canonical_roi(row: ManifestRow) -> tuple[bytes, np.ndarray]:
    try:
        payload = row.roi_path.read_bytes()
    except OSError as exc:
        raise PlainBatchError(f"cannot read ROI for {row.sample_id}") from exc
    _require(payload.startswith(_PNG_SIGNATURE), f"{row.sample_id}: ROI is not PNG")
    _require(
        _sha256_bytes(payload) == row.roi_png_sha256,
        f"{row.sample_id}: ROI PNG hash drift",
    )
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    _require(image is not None, f"{row.sample_id}: ROI PNG decode failed")
    image = np.ascontiguousarray(image)
    _require(
        canonical_roi_pixel_sha256(image) == row.roi_pixel_sha256,
        f"{row.sample_id}: decoded ROI pixel hash drift",
    )
    return payload, image


@dataclass(frozen=True)
class _InternalSpec:
    arm: str
    parent_path: Path
    parent_sha256: str
    parent_state_sha256: str
    gain_path: Path
    gain_sha256: str


def _run_path(stage: str) -> Path:
    return _SEALED_ROOT / "runs" / stage / "terminal.pt"


_PARENT_20: Final[tuple[Path, str, str]] = (
    _run_path("solver_parent_seed_20262020"),
    "4cd14d1995e5f96e33a791db01e750079a42e9e0fd9365c4591f54e105e6d02d",
    "2a80efdf78e03355049ff20af18cc1fb3dcd721583ab9ab9b2f9c4bab6bbda01",
)
_INTERNAL_SPECS: Final[dict[str, _InternalSpec]] = {
    "full_seed_20262020": _InternalSpec(
        "full_gated",
        *_PARENT_20,
        _run_path("gain_full_gated_parent_20262020_seed_20262030"),
        "bfee0897067b694e511a0bfe060cf66a9cfd771c00d7be2bb2b9771e525e60a1",
    ),
    "full_seed_20262021": _InternalSpec(
        "full_gated",
        _run_path("solver_parent_seed_20262021"),
        "544198addbf5eeb6860c86527a25118ad744e4dcd6051e381f644b6ce372b169",
        "6def668540d9862c65f0058d34bceac41ce720e6a28239bf4f13dee81da4b2d3",
        _run_path("gain_full_gated_parent_20262021_seed_20262031"),
        "479c115c600c3ec9c0a1dfdf7d24841e4bea04c6775bbe0f5a0d5b53deb95601",
    ),
    "full_seed_20262022": _InternalSpec(
        "full_gated",
        _run_path("solver_parent_seed_20262022"),
        "4a0c3e8166eba51cd6fd7505c6529e7726ac0590524794b10a38e808cd0c7204",
        "c53f3f61c904df6a15021436d0f1ae8f49b659594b432b8bb9cb2e40b88b925d",
        _run_path("gain_full_gated_parent_20262022_seed_20262032"),
        "6dae090867e89cc06bf33a382f4f9812ee42133f14e4b49ebab4316cd8bfd106",
    ),
    "control_no_pepd_residual_seed_20262020": _InternalSpec(
        "no_pepd_residual",
        *_PARENT_20,
        _run_path("gain_no_pepd_residual_parent_20262020_seed_20262030"),
        "b73e7a8fb7048fd7489473ad4e14cf5940b48ddfd87fbb83aaa02e1e27f37ca8",
    ),
    "control_no_mask_residual_seed_20262020": _InternalSpec(
        "no_mask_residual",
        *_PARENT_20,
        _run_path("gain_no_mask_residual_parent_20262020_seed_20262030"),
        "a18d00701e720891bce6fae462841d7336b5b107f8d8d2d289bee388af83b83c",
    ),
    "control_solver_core_seed_20262020": _InternalSpec(
        "solver_core",
        *_PARENT_20,
        _run_path("gain_solver_core_parent_20262020_seed_20262030"),
        "0e336c5742d68f2c8a8ed04ae84f30e8f18dee0532c346c1bc6535a1e77197a6",
    ),
}


def _finite_progress(value: Any) -> float:
    if isinstance(value, bool):
        raise ModelPredictionFailure("boolean_prediction")
    try:
        progress = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelPredictionFailure("non_numeric_prediction") from exc
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ModelPredictionFailure("prediction_outside_closed_unit_interval")
    return progress


def _configure_cuda_runtime() -> None:
    """Apply the shared deterministic CUDA policy before any model import."""

    expected_workspace = ":4096:8"
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    _require(
        existing_workspace in (None, expected_workspace),
        "CUBLAS_WORKSPACE_CONFIG conflicts with the paper runtime",
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
    os.environ["YOLO_OFFLINE"] = "true"

    import torch

    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() >= 1, "CUDA device 0 is unavailable")
    _require(
        str(torch.cuda.get_device_name(0)) == "NVIDIA GeForce RTX 4060",
        "paper batch requires NVIDIA GeForce RTX 4060 on cuda:0",
    )
    torch.cuda.set_device(torch.device("cuda:0"))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20260807)
    torch.cuda.manual_seed_all(20260807)


def _build_internal_predictor(method: str, device: str) -> Callable[[np.ndarray], float]:
    import torch

    from experiments.cagh_v5_solver_gated_residual import CAGHV5SolverGatedResidual
    from experiments.pivot_direction_fallback import normalized_rgb_tensor
    from experiments.train_cagh_scalemark_reference_probe_v5 import canonical_tight_roi

    spec = _INTERNAL_SPECS[method]
    _require(device == "cuda:0", "paper batch internal models require cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(_sha256_file(spec.parent_path) == spec.parent_sha256, "parent checkpoint hash drift")
    _require(_sha256_file(spec.gain_path) == spec.gain_sha256, "gain checkpoint hash drift")
    parent = torch.load(spec.parent_path, map_location="cpu", weights_only=False)
    gain = torch.load(spec.gain_path, map_location="cpu", weights_only=False)
    _require(isinstance(parent, Mapping) and isinstance(gain, Mapping), "checkpoint is not a mapping")
    _require(
        parent.get("model_state_sha256") == spec.parent_state_sha256
        and isinstance(parent.get("model_state"), Mapping),
        "parent state binding drift",
    )
    gain_state = gain.get("gain_state")
    _require(
        gain.get("arm") == spec.arm
        and isinstance(gain_state, Mapping)
        and set(gain_state) == {"pepd_residual_gain", "mask_residual_gain"},
        "gain state binding drift",
    )
    model = CAGHV5SolverGatedResidual(progress_bins=72, dropout=0.10)
    model.load_parent_state_dict(parent["model_state"])
    with torch.no_grad():
        model.pepd_residual_gain.fill_(float(gain_state["pepd_residual_gain"]))
        model.mask_residual_gain.fill_(float(gain_state["mask_residual_gain"]))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()

    def predict(image_bgr: np.ndarray) -> float:
        crop, original_to_normalized, normalized_to_isotropic, _ = canonical_tight_roi(
            image_bgr, None
        )
        image = normalized_rgb_tensor(crop)[None].to(torch_device)
        final_to_isotropic = torch.from_numpy(
            normalized_to_isotropic.astype(np.float32)
        )[None].to(torch_device)
        crop_affine = torch.from_numpy(
            original_to_normalized[:2].astype(np.float32)
        )[None].to(torch_device)
        with torch.inference_mode():
            output = model(image, final_to_isotropic, crop_affine, arm=spec.arm)
        if int(output.expected_progress.shape[0]) != 1 or not bool(
            output.valid[0].detach().cpu()
        ):
            raise ModelPredictionFailure("solver_geometry_invalid")
        return _finite_progress(output.expected_progress[0].detach().cpu().item())

    return predict


def _pin_yolo_cuda0(point_detector: Any) -> None:
    yolo = getattr(point_detector, "fire_detection_model", None)
    _require(yolo is not None and callable(getattr(yolo, "to", None)), "YOLO detector is incomplete")
    yolo.to("cuda:0")
    original_predict = yolo.predict

    def predict_on_cuda0(_instance: Any, image: Any, confidence: Any = None) -> Any:
        kwargs: dict[str, Any] = {"verbose": False, "device": "cuda:0"}
        if confidence is not None:
            kwargs["conf"] = float(confidence)
        return original_predict(image, **kwargs)

    point_detector._predict = types.MethodType(predict_on_cuda0, point_detector)


def _reported_failure(record: Mapping[str, Any]) -> str:
    direct = record.get("failure_code")
    if isinstance(direct, str) and direct:
        return direct
    nested = record.get("failure")
    if isinstance(nested, Mapping) and isinstance(nested.get("code"), str):
        return str(nested["code"])
    return "model_reported_failure"


def _build_vdn_predictor(_method: str, device: str) -> Callable[[np.ndarray], float]:
    from experiments import vdn_seed20_terminal_matched_baseline as vdn

    _require(device == "cuda:0", "paper batch VDN requires cuda:0")
    provider = vdn.build_seed20_terminal_matched_provider(
        device=device,
        amp_enabled=True,
    )
    _pin_yolo_cuda0(provider.reference_provider.point_detector)

    def predict(image_bgr: np.ndarray) -> float:
        record = provider.predict(
            np.ascontiguousarray(image_bgr).copy(),
            input_is_canonical_meter_roi=True,
        )
        if not isinstance(record, Mapping) or record.get("status") is not True:
            raise ModelPredictionFailure(
                _reported_failure(record if isinstance(record, Mapping) else {})
            )
        return _finite_progress(record.get("prediction_progress"))

    return predict


def _build_transformer_predictor(
    _method: str, device: str
) -> Callable[[np.ndarray], float]:
    # The checked-in legacy runtime uses bare ``dataloader``/``vitTranforms``
    # imports.  Adding its own source directory is the normal legacy bootstrap,
    # not a custom importer.
    angle_root = _PROJECT_ROOT / "utils" / "angleDetect"
    angle_text = str(angle_root)
    if angle_text not in sys.path:
        sys.path.insert(0, angle_text)

    from experiments.v5_unified_legacy_adapters import (
        FrozenCanonicalROILegacyRuntime,
        OriginalTransformerAutomaticReferenceAdapter,
    )

    _require(device == "cuda:0", "paper batch Transformer requires cuda:0")
    pointer_path = _PROJECT_ROOT / "utils/angleDetect/pointerSeg/resultSeg/best.pt"
    transformer_path = _PROJECT_ROOT / "utils/angleDetect/vitTranforms/result/best.pt"
    reference_path = (
        _PROJECT_ROOT / "utils/angleDetect/yoloDetection/result/yolo_pointbest.pt"
    )
    expected = {
        pointer_path: "27a48bd42cdce19949f2b2fc61746b801aec206d98320e338571d0c0545a34f8",
        transformer_path: "e7592c7f0d782674b645d4b331845e313390d0b1aa756a637de7d798cab56360",
        reference_path: "2cb5c2523e364063ccdfd5c047390f17986ebcb622cef09d8604dfdaf038bdd6",
    }
    for path, digest in expected.items():
        _require(_sha256_file(path) == digest, f"legacy artifact hash drift: {path.name}")
    runtime = FrozenCanonicalROILegacyRuntime.from_checkpoints(
        pointer_segmentation=pointer_path,
        original_transformer=transformer_path,
        reference_detector=reference_path,
        device=device,
    )
    _pin_yolo_cuda0(runtime.runtime.pointerDetect)
    provider = OriginalTransformerAutomaticReferenceAdapter(runtime)

    def predict(image_bgr: np.ndarray) -> float:
        record = provider.predict(
            np.ascontiguousarray(image_bgr).copy(),
            input_is_canonical_meter_roi=True,
        )
        if not isinstance(record, Mapping) or record.get("status") != "ok":
            raise ModelPredictionFailure(
                _reported_failure(record if isinstance(record, Mapping) else {})
            )
        return _finite_progress(record.get("progress"))

    return predict


_BUILDERS: Final[dict[str, Callable[[str, str], Callable[[np.ndarray], float]]]] = {
    **{method: _build_internal_predictor for method in _INTERNAL_SPECS},
    "vdn_official200_terminal_seed20": _build_vdn_predictor,
    "original_transformer_legacy_auto_reference": _build_transformer_predictor,
}


def build_predictor(method: str, *, device: str = "cuda:0") -> Callable[[np.ndarray], float]:
    _require(method in METHODS, f"unknown method: {method}")
    _require(tuple(_BUILDERS) == METHODS, "paper method registry drift")
    _require(device == "cuda:0", "paper batch supports only cuda:0")
    _configure_cuda_runtime()
    return _BUILDERS[method](method, device)


def _result_row(
    *,
    source: ManifestRow,
    method: str,
    condition: str,
    condition_pixel_sha256: str,
    progress: float | None,
    failure_code: str | None,
) -> dict[str, Any]:
    passed = progress is not None and failure_code is None
    row = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "sample_id": source.sample_id,
        "method": method,
        "condition": condition,
        "robustness_seed": ROBUSTNESS_SEED,
        "status": "pass" if passed else "fail",
        "normalized_progress": progress if passed else None,
        "failure_code": None if passed else failure_code,
        "roi_png_sha256": source.roi_png_sha256,
        "roi_pixel_sha256": source.roi_pixel_sha256,
        "condition_pixel_sha256": condition_pixel_sha256,
    }
    _require(set(row) == set(OUTPUT_KEYS), "output schema drift")
    return row


def run_batch(
    *,
    method: str,
    manifest_path: Path,
    output_path: Path,
    device: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
    predictor_factory: Callable[..., Callable[[np.ndarray], float]] = build_predictor,
) -> int:
    _require(method in METHODS, f"unknown method: {method}")
    _require(
        robustness_degradations.degradation_names(include_clean=True) == CONDITIONS,
        "robustness condition roster drift",
    )
    selected_conditions = tuple(conditions)
    _require(bool(selected_conditions), "at least one condition is required")
    _require(
        len(selected_conditions) == len(set(selected_conditions))
        and set(selected_conditions) <= set(CONDITIONS),
        "selected condition roster is invalid",
    )
    manifest = Path(manifest_path).resolve()
    output = Path(output_path).resolve()
    _require(manifest != output, "output cannot overwrite the input manifest")
    rows = load_manifest(manifest)
    _require(all(row.roi_path != output for row in rows), "output cannot overwrite an ROI")
    predictor = predictor_factory(method, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            for condition in selected_conditions:
                conditioned, _metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                conditioned = np.ascontiguousarray(conditioned)
                condition_hash = canonical_roi_pixel_sha256(conditioned)
                progress: float | None = None
                failure_code: str | None = None
                try:
                    progress = _finite_progress(predictor(conditioned.copy()))
                except ModelPredictionFailure as exc:
                    failure_code = exc.code
                except Exception as exc:  # Per-image failures remain in the cohort.
                    failure_code = f"model_exception:{type(exc).__name__}"
                stream.write(
                    _canonical_json_line(
                        _result_row(
                            source=source,
                            method=method,
                            condition=condition,
                            condition_pixel_sha256=condition_hash,
                            progress=progress,
                            failure_code=failure_code,
                        )
                    ).decode("utf-8")
                )
                count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
        help="Conditions to evaluate; defaults to clean plus all five stresses.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    count = run_batch(
        method=args.method,
        manifest_path=args.manifest,
        output_path=args.output,
        device=args.device,
        conditions=args.conditions,
    )
    sys.stdout.write(
        json.dumps(
            {
                "method": args.method,
                "output": str(Path(args.output).resolve()),
                "prediction_rows": count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "METHODS",
    "ModelPredictionFailure",
    "PlainBatchError",
    "build_predictor",
    "canonical_roi_pixel_sha256",
    "load_manifest",
    "run_batch",
]
