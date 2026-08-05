"""Evaluate a SyncG-retrained official VDN as an end-to-end reading baseline.

VDN replaces only pointer direction estimation.  Meter detection, start/end
references, scale conversion, failure handling, and controlled degradations
follow the same frozen protocol used by the project's paper-facing methods.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.torch_version import TorchVersion
from tqdm import tqdm

from experiments.robustness_degradations import (
    ROBUSTNESS_PROTOCOL,
    apply_degradation,
    degradation_names,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    VDN_PROTOCOL,
    build_vdn_model,
    image_angle_from_direction,
    normalize_reference_points,
    predict_directions,
    reading_from_pointer_angle,
    reference_angles,
    sha256_file,
    sha256_source_file,
    summarize_scalar_predictions,
    vdn_tensor_from_bbox,
    verify_vdn_source,
)
from experiments.vdn_phase2_protocol import (
    PHASE2_CHECKPOINT_PROTOCOL,
    PHASE2_PROTOCOL,
)
from experiments.vdn_phase2_evaluation_plan import (
    PINNED_PUBLIC_RELEASE_INVENTORY_SHA256,
    validate_phase2_evaluation_plan,
    validate_phase2_public_preflight,
)
from experiments.verify_vdn_phase2_cohort import (
    _strict_json,
    validate_cohort_evaluation_authorization,
)


EVALUATION_PROTOCOL = "vdn_syncg_external_baseline_e2e_v1"
PHASE2_EVALUATION_PROTOCOL = "vdn_syncg_phase2_external_baseline_e2e_v1"
LEGACY_VERIFICATION_PROTOCOL = "formal_vdn_run_verification_v1"
PREDICTION_JOURNAL_PROTOCOL = "vdn_prediction_append_journal_v1"
_FORBIDDEN_EVALUATION_PATH_TOKENS = (
    "field",
    "confirmatory",
    "sealed",
)
FORMAL_LEGACY_SEEDS = (20260720, 20260721, 20260722)
DEFAULT_LEGACY_RUN_ROOT = PROJECT_DIR / "artifacts" / "runs" / "vdn_syncg"
_LEGACY_VERIFICATION_KEYS = frozenset(
    {
        "best_checkpoint_sha256",
        "best_epoch",
        "best_validation_angle_mae_degrees",
        "epochs",
        "group_overlap",
        "last_checkpoint_sha256",
        "model_state_health",
        "optimizer_steps",
        "protocol",
        "run_dir",
        "skipped_optimizer_steps",
        "summary_sha256",
        "train_groups",
        "train_samples",
        "validation_groups",
        "validation_samples",
        "vdn_source_commit",
        "verified",
        "verifier_source_sha256",
    }
)
DEFAULT_METER_WEIGHTS = (
    PROJECT_DIR
    / "utils"
    / "angleDetect"
    / "yoloDetection"
    / "result"
    / "yolo_findMeter.pt"
)
DEFAULT_POINT_WEIGHTS = (
    PROJECT_DIR
    / "utils"
    / "angleDetect"
    / "yoloDetection"
    / "result"
    / "yolo_pointbest.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--phase2-cohort-authorization",
        type=Path,
        help=(
            "required for every Phase-2 checkpoint; validated before any "
            "test/public manifest or prediction cache is opened"
        ),
    )
    parser.add_argument(
        "--phase2-public-preflight",
        type=Path,
        help=(
            "required for every Phase-2 checkpoint; its exact report hash "
            "is bound into the evaluation signature"
        ),
    )
    parser.add_argument(
        "--legacy-training-verification",
        type=Path,
        help=(
            "required instead of a Phase-2 cohort for a frozen legacy VDN "
            "run; the two external authorization modes are mutually exclusive"
        ),
    )
    parser.add_argument(
        "--vdn-source",
        type=Path,
        default=(
            PROJECT_DIR
            / "artifacts"
            / "vendor"
            / "VectorDetectionNetwork"
        ),
    )
    parser.add_argument("--shared-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--condition", choices=degradation_names(), default="clean")
    parser.add_argument("--degradation-seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--meter-detector-weights", type=Path, default=DEFAULT_METER_WEIGHTS)
    parser.add_argument("--keypoint-detector-weights", type=Path, default=DEFAULT_POINT_WEIGHTS)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _read_jsonl_bytes(
    payload: bytes,
    *,
    label: str,
) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(
                f"{label}:{line_number} is not a JSON object"
            )
        rows.append(row)
    return rows


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _portable_release_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): _portable_release_value(
                child,
                key=str(child_key),
            )
            for child_key, child in sorted(value.items())
        }
    if isinstance(value, list):
        return [_portable_release_value(child, key=key) for child in value]
    if isinstance(value, tuple):
        return [_portable_release_value(child, key=key) for child in value]
    if isinstance(value, str) and any(
        token in key.casefold()
        for token in ("path", "root", "file", "source_labels")
    ):
        return Path(value).name
    return value


def _authorize_public_image_inventory(
    rows: Sequence[dict[str, Any]],
    *,
    public_scope: str,
) -> tuple[str, dict[str, str]]:
    """Bind original public image bytes to a preregistered release digest."""

    try:
        expected = PINNED_PUBLIC_RELEASE_INVENTORY_SHA256[public_scope]
    except KeyError as exc:
        raise ValueError(f"unsupported public scope: {public_scope}") from exc
    records: list[dict[str, Any]] = []
    image_hashes: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: str(item.get("sample_id") or "")):
        sample_id = str(row.get("sample_id") or "")
        image_value = row.get("image_path")
        if not sample_id or not isinstance(image_value, str) or not image_value:
            raise ValueError(
                "public release inventory requires sample_id and image_path"
            )
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path
        resolved = image_path.resolve()
        folded_path = str(resolved).casefold()
        if any(
            token in folded_path
            for token in _FORBIDDEN_EVALUATION_PATH_TOKENS
        ):
            raise PermissionError(
                "public evaluation manifest references a forbidden scope"
            )
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        image_digest = sha256_file(resolved)
        if sample_id in image_hashes:
            raise ValueError("public release repeats a sample identifier")
        image_hashes[sample_id] = image_digest
        portable_row = {
            key: _portable_release_value(value, key=str(key))
            for key, value in sorted(row.items())
            if key != "image_path"
        }
        records.append(
            {
                "sample_id": sample_id,
                "image_name": resolved.name,
                "image_sha256": image_digest,
                "portable_manifest_row": portable_row,
            }
        )
    actual = _canonical_json_sha256(records)
    if actual != expected:
        raise ValueError(
            f"{public_scope} content inventory differs from the pinned "
            f"public release ({actual} != {expected})"
        )
    return actual, image_hashes


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _metadata_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def _summary_path(output: Path) -> Path:
    return output.with_name(output.stem + ".summary.json")


def _journal_path(output: Path) -> Path:
    return output.with_name(output.name + ".journal.jsonl")


@contextmanager
def _exclusive_output_lock(output: Path):
    """Hold a process-level lock for one formal prediction destination."""

    lock_path = output.with_name(output.name + ".writer.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        handle.close()
        raise RuntimeError(
            f"another evaluator owns the output lock: {lock_path}"
        ) from exc
    try:
        yield lock_path
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _atomic_json_no_clobber(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=f".{uuid.uuid4().hex}.tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite immutable JSON: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    def distribution_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    torch_build = torch.__config__.show()
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_build_config_sha256": hashlib.sha256(
            torch_build.encode("utf-8")
        ).hexdigest(),
        "torchvision": distribution_version("torchvision"),
        "ultralytics": distribution_version("ultralytics"),
        "pillow": distribution_version("Pillow"),
        "opencv": str(cv2.__version__),
        "numpy": str(np.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_driver_version": None,
        "cudnn": (
            int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None
        ),
        "device_type": str(device.type),
        "device_index": device.index,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "python_safe_path": bool(sys.flags.safe_path),
        "python_no_user_site": bool(sys.flags.no_user_site),
        "python_ignore_environment": bool(
            sys.flags.ignore_environment
        ),
        "python_hash_randomization": bool(
            sys.flags.hash_randomization
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    if device.type == "cuda":
        driver_version = getattr(torch.cuda, "driver_version", None)
        if callable(driver_version):
            result["cuda_driver_version"] = int(driver_version())
        else:
            internal_driver_version = getattr(
                torch._C,
                "_cuda_getDriverVersion",
                None,
            )
            if callable(internal_driver_version):
                result["cuda_driver_version"] = int(
                    internal_driver_version()
                )
        index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        result["device_index"] = index
        result["device_name"] = torch.cuda.get_device_name(index)
        result["device_capability"] = list(
            torch.cuda.get_device_capability(index)
        )
        properties = torch.cuda.get_device_properties(index)
        result["device_total_memory"] = int(properties.total_memory)
        result["device_multiprocessor_count"] = int(
            properties.multi_processor_count
        )
    else:
        result["device_name"] = platform.processor()
        result["device_capability"] = None
        result["device_total_memory"] = None
        result["device_multiprocessor_count"] = None
    return result


def _load_target_detector_class() -> type:
    """Import the detector only after authorization from the project tree."""

    distribution = importlib.metadata.distribution("ultralytics")
    expected_ultralytics_root = Path(
        distribution.locate_file("ultralytics")
    ).resolve()
    ultralytics_spec = importlib.util.find_spec("ultralytics")
    if (
        ultralytics_spec is None
        or ultralytics_spec.origin is None
        or not Path(ultralytics_spec.origin).resolve().is_relative_to(
            expected_ultralytics_root
        )
    ):
        raise ImportError(
            "ultralytics does not resolve to the installed distribution"
        )
    module_names = (
        "utils",
        "utils.angleDetect",
        "utils.angleDetect.yoloDetection",
        "utils.angleDetect.yoloDetection.pointGet",
        "utils.angleDetect.yoloDetection.yoloDectect",
    )
    for name in module_names:
        loaded = sys.modules.get(name)
        if loaded is None:
            continue
        source = getattr(loaded, "__file__", None)
        search_paths = getattr(loaded, "__path__", ())
        candidates = (
            [Path(source).resolve()]
            if source is not None
            else [Path(path).resolve() for path in search_paths]
        )
        if not candidates or not all(
            candidate.is_relative_to(PROJECT_DIR) for candidate in candidates
        ):
            raise ImportError(
                f"refusing preloaded detector package outside PROJECT_DIR: {name}"
            )

    project_entry = str(PROJECT_DIR)
    sys.path.insert(0, project_entry)
    try:
        module = importlib.import_module(
            "utils.angleDetect.yoloDetection.yoloDectect"
        )
        point_module = importlib.import_module(
            "utils.angleDetect.yoloDetection.pointGet"
        )
    finally:
        if sys.path and sys.path[0] == project_entry:
            sys.path.pop(0)

    expected_module = (
        PROJECT_DIR
        / "utils"
        / "angleDetect"
        / "yoloDetection"
        / "yoloDectect.py"
    ).resolve()
    expected_point = expected_module.with_name("pointGet.py")
    if (
        Path(module.__file__).resolve() != expected_module
        or Path(point_module.__file__).resolve() != expected_point
    ):
        raise ImportError("detector modules were not imported from PROJECT_DIR")
    detector_class = getattr(module, "targetDetectModel", None)
    if not isinstance(detector_class, type):
        raise ImportError("formal detector class is missing")
    return detector_class


def _configure_evaluation_determinism(training_protocol: str) -> None:
    if training_protocol == PHASE2_PROTOCOL:
        _validate_phase2_process_isolation()
        if os.environ.get("PYTHONHASHSEED") != "20260722":
            raise RuntimeError(
                "formal Phase-2 evaluation requires PYTHONHASHSEED=20260722 "
                "before Python starts"
            )
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError(
                "formal Phase-2 evaluation requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")


def _validate_phase2_process_isolation() -> None:
    if int(sys.flags.safe_path) != 1:
        raise RuntimeError(
            "formal Phase-2 evaluation requires Python safe-path mode (-P)"
        )
    if int(sys.flags.no_user_site) != 1:
        raise RuntimeError(
            "formal Phase-2 evaluation requires the user site disabled (-s)"
        )
    if int(sys.flags.ignore_environment) != 0:
        raise RuntimeError(
            "formal Phase-2 evaluation must honor the pinned PYTHONHASHSEED"
        )
    if os.environ.get("PYTHONPATH"):
        raise RuntimeError(
            "formal Phase-2 evaluation forbids an inherited PYTHONPATH"
        )
    if os.environ.get("PYTHONHASHSEED") != "20260722":
        raise RuntimeError(
            "formal Phase-2 evaluation requires PYTHONHASHSEED=20260722"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal Phase-2 evaluation requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )


def _safe_load_authorized_checkpoint(
    checkpoint_path: Path,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    """Load exactly the bytes whose digest was externally authorized."""

    payload = Path(checkpoint_path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != authorization.get("checkpoint_sha256"):
        raise ValueError("checkpoint changed after external authorization")
    with torch.serialization.safe_globals([TorchVersion]):
        value = torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(value, dict):
        raise ValueError("authorized checkpoint is not an object")
    return value


def _read_pinned_bytes(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> bytes:
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{label} changed between plan authorization and snapshot"
        )
    return payload


def _snapshot_phase2_inputs(
    plan_identity: dict[str, Any],
) -> dict[str, bytes]:
    """Read every mutable plan file once and bind the consumed bytes."""

    fields = {
        "manifest": ("manifest_path", "manifest_sha256"),
        "manifest_protocol": (
            "manifest_protocol_path",
            "manifest_protocol_sha256",
        ),
        "shared_predictions": (
            "shared_predictions_path",
            "shared_predictions_sha256",
        ),
        "shared_predictions_metadata": (
            "shared_predictions_metadata_path",
            "shared_predictions_metadata_sha256",
        ),
        "meter_detector_weights": (
            "meter_detector_weights_path",
            "meter_detector_weights_sha256",
        ),
        "keypoint_detector_weights": (
            "keypoint_detector_weights_path",
            "keypoint_detector_weights_sha256",
        ),
    }
    return {
        name: _read_pinned_bytes(
            Path(str(plan_identity[path_key])),
            str(plan_identity[digest_key]),
            label=name,
        )
        for name, (path_key, digest_key) in fields.items()
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_shared_predictions(
    path: Path | None,
    *,
    manifest: Path,
    rows: Sequence[dict[str, Any]],
    condition: str,
    degradation_seed: int,
    meter_weights: Path,
    point_weights: Path,
    authorized_inputs: dict[str, bytes] | None = None,
    plan_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    metadata_path = _metadata_path(path)
    if authorized_inputs is None:
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"shared predictions or metadata missing: {path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        if plan_identity is None:
            raise ValueError("authorized shared bytes require a plan identity")
        metadata = json.loads(
            authorized_inputs["shared_predictions_metadata"].decode(
                "utf-8-sig"
            )
        )
    if not isinstance(metadata, dict):
        raise ValueError("shared prediction metadata is not an object")
    signature = metadata.get("signature") or {}
    manifest_sha256 = (
        str(plan_identity["manifest_sha256"])
        if plan_identity is not None
        else sha256_file(manifest)
    )
    if signature.get("manifest_sha256") != manifest_sha256:
        raise ValueError("shared predictions use a different manifest")
    protocol_path = _protocol_path(manifest)
    expected_manifest_protocol = (
        str(plan_identity["manifest_protocol_sha256"])
        if plan_identity is not None
        else (
            sha256_file(protocol_path)
            if protocol_path.is_file()
            else None
        )
    )
    if signature.get("manifest_protocol_sha256") != expected_manifest_protocol:
        raise ValueError("shared predictions use a different manifest protocol")
    if signature.get("correction_mode") != "off":
        raise ValueError("VDN comparison requires correction_mode=off shared references")
    degradation = signature.get("input_degradation") or {}
    expected_degradation_source = sha256_file(
        PROJECT_DIR / "experiments" / "robustness_degradations.py"
    )
    recorded_degradation_source = signature.get("input_degradation_source_sha256")
    if degradation:
        if degradation.get("protocol") != ROBUSTNESS_PROTOCOL:
            raise ValueError("shared predictions use a different degradation protocol")
        if recorded_degradation_source != expected_degradation_source:
            raise ValueError("shared predictions use different degradation source code")
    elif condition != "clean" or recorded_degradation_source is not None:
        raise ValueError("only clean legacy caches may omit the degradation signature")
    cached_condition = degradation.get("condition", "clean")
    if cached_condition != condition:
        raise ValueError(
            f"shared prediction condition is {cached_condition}, expected {condition}"
        )
    cached_seed = int(degradation.get("seed", degradation_seed))
    if condition != "clean" and cached_seed != degradation_seed:
        raise ValueError("shared predictions use a different degradation seed")
    weight_hashes = signature.get("weights_sha256") or {}
    expected_hashes = {
        "meter_detector": (
            str(plan_identity["meter_detector_weights_sha256"])
            if plan_identity is not None
            else sha256_file(meter_weights)
        ),
        "keypoint_detector": (
            str(plan_identity["keypoint_detector_weights_sha256"])
            if plan_identity is not None
            else sha256_file(point_weights)
        ),
    }
    for name, expected in expected_hashes.items():
        if weight_hashes.get(name) != expected:
            raise ValueError(f"shared predictions use different {name} weights")

    shared_rows = (
        _read_jsonl(path)
        if authorized_inputs is None
        else _read_jsonl_bytes(
            authorized_inputs["shared_predictions"],
            label=str(path),
        )
    )
    by_id = {str(row.get("sample_id")): row for row in shared_rows}
    if len(by_id) != len(shared_rows):
        raise ValueError("shared predictions contain duplicate sample identifiers")
    missing = [str(row.get("sample_id")) for row in rows if str(row.get("sample_id")) not in by_id]
    if missing:
        raise ValueError(f"shared predictions miss {len(missing)} selected samples")
    return by_id, metadata


def _shared_reference(
    row: dict[str, Any] | None,
) -> tuple[float, float, str] | None:
    if not row:
        return None
    features = row.get("features") or {}
    start_angle = _finite_float(features.get("startAngle"))
    range_angle = _finite_float(features.get("disAngle"))
    if start_angle is None or range_angle is None or abs(range_angle) <= 1e-8:
        return None
    return start_angle, range_angle, str(row.get("branch") or "shared_unknown")


def _fallback_reference(
    point_detector: Any,
    crop: np.ndarray,
) -> tuple[float, float, str]:
    # Match zeroShotMeter exactly: its crop is converted BGR->RGB before PIL,
    # then the returned RGB array is passed to OpenCV using BGR2GRAY.
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    detector_input = np.stack((gray,) * 3, axis=-1)
    center_end, _ = point_detector.center_find(detector_input, classId=1)
    center_start, _ = point_detector.center_find(detector_input, classId=2)
    center_start, center_end = normalize_reference_points(
        center_start,
        center_end,
        image_width=crop.shape[1],
    )
    return reference_angles(crop.shape, center_start, center_end)


def _xyxy_from_detector_box(box: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(box, dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError(f"unexpected meter box shape: {array.shape}")
    x1, y1 = map(float, array[0])
    x2, y2 = map(float, array[2])
    if x2 <= x1 or y2 <= y1:
        raise ValueError("meter detector returned an empty box")
    return x1, y1, x2, y2


def _pointer_points(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    for item in metadata.get("keypoints") or []:
        if str(item.get("type") or "").strip().lower() != "pointer":
            continue
        tip = item.get("outside_kp")
        tail = item.get("origin_kp")
        if isinstance(tip, Sequence) and isinstance(tail, Sequence):
            return (
                np.asarray(tip[:2], dtype=np.float64),
                np.asarray(tail[:2], dtype=np.float64),
            )
    return None


def _target_direction(
    row: dict[str, Any],
    degradation: dict[str, Any],
) -> np.ndarray | None:
    points = _pointer_points(row.get("metadata") or {})
    if points is None:
        return None
    tip, tail = points
    perspective = degradation.get("perspective") or {}
    homography = perspective.get("homography")
    if homography is not None:
        stacked = np.asarray([[tail, tip]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(
            stacked,
            np.asarray(homography, dtype=np.float64),
        )[0]
        tail, tip = transformed[0], transformed[1]
    direction = np.asarray(tip, dtype=np.float64) - np.asarray(tail, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-8 else None


def _direction_error(predicted: np.ndarray, target: np.ndarray | None) -> float | None:
    if target is None:
        return None
    cosine = float(np.clip(np.dot(predicted, target), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _base_result(row: dict[str, Any], degradation: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row.get("sample_id")),
        "group_id": str(row.get("group_id") or row.get("meter_id") or row.get("sample_id")),
        "meter_id": row.get("meter_id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "image_path": row.get("image_path"),
        "input_image_sha256": row.get(
            "_authorized_input_image_sha256"
        ),
        "ground_truth": float(row["ground_truth"]),
        "scale_start": float(row["scale_start"]),
        "scale_end": float(row["scale_end"]),
        "metadata": row.get("metadata") or {},
        "degradation": degradation,
        "status": False,
        "prediction": None,
        "progress": None,
        "pointer_angle": None,
        "direction": None,
        "direction_angle_error_degrees": None,
        "heatmap_peak": None,
        "error_code": None,
        "error_message": None,
    }


def _failure_result(
    row: dict[str, Any],
    degradation: dict[str, Any],
    *,
    code: str,
    message: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    result = _base_result(row, degradation)
    result.update(
        {
            "error_code": code,
            "error_message": message,
            "runtime_seconds": float(runtime_seconds),
        }
    )
    return result


def _append_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class _PredictionJournal:
    """Append predictions with a verified cumulative prefix-hash journal."""

    _KEYS = frozenset(
        {
            "protocol",
            "sequence",
            "rows_added",
            "total_rows",
            "byte_offset",
            "prefix_sha256",
            "previous_record_sha256",
            "record_sha256",
        }
    )

    def __init__(
        self,
        output: Path,
        journal: Path,
        *,
        resume: bool,
        recover_uncommitted_tail: bool = False,
    ) -> None:
        self.output = Path(output)
        self.journal = Path(journal)
        self._hasher = hashlib.sha256()
        self.sequence = 0
        self.total_rows = 0
        self.byte_offset = 0
        self.previous_record_sha256: str | None = None
        self.recovered_output_tail_bytes = 0
        self.recovered_journal_tail_bytes = 0
        if resume:
            self._validate_existing(
                recover_uncommitted_tail=recover_uncommitted_tail
            )
        elif self.output.read_bytes() or self.journal.read_bytes():
            raise RuntimeError("new prediction/journal files are not empty")

    @staticmethod
    def _record_sha256(record: dict[str, Any]) -> str:
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        payload = json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _strict_line(line: str, *, line_number: int) -> dict[str, Any]:
        def reject_constant(value: str) -> None:
            raise ValueError(
                f"prediction journal line {line_number} has {value}"
            )

        def reject_duplicates(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(
                        "prediction journal line "
                        f"{line_number} repeats {key!r}"
                    )
                value[key] = item
            return value

        value = json.loads(
            line,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
        if not isinstance(value, dict):
            raise ValueError(
                f"prediction journal line {line_number} is not an object"
            )
        return value

    @staticmethod
    def _truncate_fsync(path: Path, size: int) -> None:
        with path.open("r+b") as handle:
            handle.truncate(size)
            handle.flush()
            os.fsync(handle.fileno())

    def _validate_existing(
        self,
        *,
        recover_uncommitted_tail: bool,
    ) -> None:
        output_bytes = self.output.read_bytes()
        journal_bytes = self.journal.read_bytes()
        journal_committed_end = len(journal_bytes)
        if journal_bytes and not journal_bytes.endswith(b"\n"):
            if not recover_uncommitted_tail:
                raise ValueError(
                    "prediction journal has an uncommitted trailing record"
                )
            last_newline = journal_bytes.rfind(b"\n")
            journal_committed_end = (
                last_newline + 1 if last_newline >= 0 else 0
            )
        committed_journal = journal_bytes[:journal_committed_end]
        records: list[dict[str, Any]] = []
        try:
            journal_text = committed_journal.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "prediction journal committed prefix is not UTF-8"
            ) from exc
        for line_number, line in enumerate(
            io.StringIO(journal_text),
            start=1,
        ):
            if not line.strip():
                raise ValueError(
                    f"prediction journal line {line_number} is empty"
                )
            records.append(
                self._strict_line(line, line_number=line_number)
            )

        previous_offset = 0
        previous_rows = 0
        previous_digest: str | None = None
        for sequence, record in enumerate(records, start=1):
            if set(record) != self._KEYS:
                raise ValueError(
                    f"prediction journal record {sequence} schema drifted"
                )
            rows_added = record.get("rows_added")
            total_rows = record.get("total_rows")
            byte_offset = record.get("byte_offset")
            if (
                record.get("protocol") != PREDICTION_JOURNAL_PROTOCOL
                or type(record.get("sequence")) is not int
                or record["sequence"] != sequence
                or type(rows_added) is not int
                or rows_added <= 0
                or type(total_rows) is not int
                or total_rows != previous_rows + rows_added
                or type(byte_offset) is not int
                or not previous_offset < byte_offset <= len(output_bytes)
                or record.get("previous_record_sha256") != previous_digest
            ):
                raise ValueError(
                    f"prediction journal record {sequence} ledger drifted"
                )
            chunk = output_bytes[previous_offset:byte_offset]
            if not chunk.endswith(b"\n") or chunk.count(b"\n") != rows_added:
                raise ValueError(
                    f"prediction journal record {sequence} row boundary drifted"
                )
            self._hasher.update(chunk)
            if record.get("prefix_sha256") != self._hasher.hexdigest():
                raise ValueError(
                    f"prediction journal record {sequence} prefix hash drifted"
                )
            record_digest = self._record_sha256(record)
            if record.get("record_sha256") != record_digest:
                raise ValueError(
                    f"prediction journal record {sequence} digest drifted"
                )
            previous_offset = byte_offset
            previous_rows = total_rows
            previous_digest = record_digest

        if previous_offset != len(output_bytes):
            if not recover_uncommitted_tail:
                raise ValueError(
                    "prediction output has bytes not committed by the journal"
                )
            self.recovered_output_tail_bytes = (
                len(output_bytes) - previous_offset
            )
        committed_output = output_bytes[:previous_offset]
        if committed_output.count(b"\n") != previous_rows:
            raise ValueError("prediction output row count differs from journal")
        self.recovered_journal_tail_bytes = (
            len(journal_bytes) - journal_committed_end
        )
        if self.recovered_output_tail_bytes:
            self._truncate_fsync(self.output, previous_offset)
        if self.recovered_journal_tail_bytes:
            self._truncate_fsync(self.journal, journal_committed_end)
        self.sequence = len(records)
        self.total_rows = previous_rows
        self.byte_offset = previous_offset
        self.previous_record_sha256 = previous_digest

    def append(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        payload = b"".join(
            (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for row in rows
        )
        with self.output.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._hasher.update(payload)
        record = {
            "protocol": PREDICTION_JOURNAL_PROTOCOL,
            "sequence": self.sequence + 1,
            "rows_added": len(rows),
            "total_rows": self.total_rows + len(rows),
            "byte_offset": self.byte_offset + len(payload),
            "prefix_sha256": self._hasher.hexdigest(),
            "previous_record_sha256": self.previous_record_sha256,
        }
        record["record_sha256"] = self._record_sha256(record)
        encoded_record = (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        with self.journal.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded_record)
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence += 1
        self.total_rows += len(rows)
        self.byte_offset += len(payload)
        self.previous_record_sha256 = record["record_sha256"]


def _component_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    available = [
        float(row["direction_angle_error_degrees"])
        for row in rows
        if row.get("direction_angle_error_degrees") is not None
    ]
    eligible = sum(_pointer_points(row.get("metadata") or {}) is not None for row in rows)
    if eligible == 0:
        return None
    errors = np.asarray(available, dtype=np.float64)
    return {
        "eligible_samples": eligible,
        "successful_directions": len(available),
        "coverage": len(available) / eligible,
        "angle_mae_degrees_success_only": float(np.mean(errors)) if len(errors) else None,
        "angle_median_degrees_success_only": float(np.median(errors)) if len(errors) else None,
        "angle_acc_1deg_all": float(np.sum(errors <= 1.0) / eligible),
        "angle_acc_3deg_all": float(np.sum(errors <= 3.0) / eligible),
        "angle_acc_5deg_all": float(np.sum(errors <= 5.0) / eligible),
    }


def _dialbench_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if abs(float(row["scale_end"]) - float(row["scale_start"])) > 1e-12
    ]
    successful = [row for row in eligible if row.get("prediction") is not None]
    reference_errors = [
        abs(float(row["prediction"]) - float(row["ground_truth"]))
        / abs(float(row["scale_end"]) - float(row["scale_start"]))
        for row in successful
    ]
    relative_eligible = [
        row for row in eligible if abs(float(row["ground_truth"])) > 1e-12
    ]
    relative_successful = [
        row for row in relative_eligible if row.get("prediction") is not None
    ]
    relative_errors = [
        abs(float(row["prediction"]) - float(row["ground_truth"]))
        / abs(float(row["ground_truth"]))
        for row in relative_successful
    ]
    return {
        "eligible_samples": len(eligible),
        "successful_samples": len(successful),
        "ref_successful": (
            float(np.mean(reference_errors)) if reference_errors else None
        ),
        "relative_samples": len(relative_eligible),
        "relative_successful": len(relative_successful),
        "rel_successful": (
            float(np.mean(relative_errors)) if relative_errors else None
        ),
        "acc_epsilon_ref_le_1pct_e2e": (
            sum(error <= 0.01 for error in reference_errors) / len(eligible)
            if eligible
            else None
        ),
        "acc_theta_rel_lt_5pct_e2e": (
            sum(error < 0.05 for error in relative_errors) / len(relative_eligible)
            if relative_eligible
            else None
        ),
    }


def _subgroup_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    memberships: dict[str, dict[str, list[dict[str, Any]]]] = {
        "meter_id": defaultdict(list),
        "environment_condition": defaultdict(list),
    }
    for row in rows:
        memberships["meter_id"][str(row.get("meter_id") or "unknown")].append(row)
        conditions = (row.get("metadata") or {}).get("environment_conditions") or []
        if isinstance(conditions, str):
            conditions = [conditions]
        for condition in conditions:
            memberships["environment_condition"][str(condition)].append(row)
    result: dict[str, Any] = {}
    for kind, values in memberships.items():
        result[kind] = {
            name: summarize_scalar_predictions(
                subset,
                bootstrap_iterations=0,
                seed=0,
            )
            for name, subset in sorted(values.items())
        }
    return result


def _run_signature(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    shared_metadata: dict[str, Any] | None,
    *,
    training_protocol: str,
    training_authorization: dict[str, Any],
    phase2_evaluation_plan: dict[str, Any] | None,
    phase2_public_preflight: dict[str, Any] | None,
    public_release_inventory_sha256: str | None,
    runtime_environment: dict[str, Any],
    amp_enabled: bool,
) -> dict[str, Any]:
    protocol_path = _protocol_path(args.manifest)
    evaluation_protocol = (
        PHASE2_EVALUATION_PROTOCOL
        if training_protocol == PHASE2_PROTOCOL
        else EVALUATION_PROTOCOL
    )
    planned = phase2_evaluation_plan or {}
    legacy_source_identity = {
        "adapter": sha256_file(
            PROJECT_DIR / "experiments" / "vdn_baseline.py"
        ),
        "evaluation": sha256_file(Path(__file__).resolve()),
        "degradation": sha256_file(
            PROJECT_DIR / "experiments" / "robustness_degradations.py"
        ),
        "meter_detector": sha256_source_file(
            PROJECT_DIR
            / "utils"
            / "angleDetect"
            / "yoloDetection"
            / "yoloDectect.py"
        ),
        "reference_point_geometry": sha256_source_file(
            PROJECT_DIR
            / "utils"
            / "angleDetect"
            / "yoloDetection"
            / "pointGet.py"
        ),
        "isolated_bootstrap": sha256_source_file(
            PROJECT_DIR
            / "experiments"
            / "run_vdn_phase2_evaluation_isolated.py"
        ),
        "phase2_evaluation_supervisor": sha256_source_file(
            PROJECT_DIR
            / "experiments"
            / "run_vdn_phase2_evaluations.ps1"
        ),
    }
    execution_source_identity = (
        planned.get("execution_source_identity") if planned else None
    )
    return {
        "protocol": evaluation_protocol,
        "training_protocol": training_protocol,
        "vdn_source_commit": verify_vdn_source(args.vdn_source),
        "checkpoint_sha256": training_authorization["checkpoint_sha256"],
        "checkpoint_training_signature": checkpoint.get("signature"),
        "training_authorization": training_authorization,
        "phase2_cohort_authorization": (
            training_authorization
            if training_protocol == PHASE2_PROTOCOL
            else None
        ),
        "phase2_evaluation_plan": phase2_evaluation_plan,
        "phase2_public_preflight": phase2_public_preflight,
        "evaluation_execution_source_identity": (
            execution_source_identity
        ),
        "public_release_inventory_sha256": (
            public_release_inventory_sha256
        ),
        "manifest_sha256": (
            planned["manifest_sha256"]
            if planned
            else sha256_file(args.manifest)
        ),
        "manifest_protocol_sha256": (
            planned["manifest_protocol_sha256"]
            if planned
            else (
                sha256_file(protocol_path)
                if protocol_path.is_file()
                else None
            )
        ),
        "condition": args.condition,
        "degradation_protocol": ROBUSTNESS_PROTOCOL,
        "degradation_seed": int(args.degradation_seed),
        "meter_detector_weights_sha256": (
            planned["meter_detector_weights_sha256"]
            if planned
            else sha256_file(args.meter_detector_weights)
        ),
        "keypoint_detector_weights_sha256": (
            planned["keypoint_detector_weights_sha256"]
            if planned
            else sha256_file(args.keypoint_detector_weights)
        ),
        "shared_predictions_sha256": (
            planned["shared_predictions_sha256"]
            if planned
            else (
                sha256_file(args.shared_predictions)
                if args.shared_predictions
                else None
            )
        ),
        "shared_predictions_metadata_sha256": (
            planned["shared_predictions_metadata_sha256"]
            if planned
            else (
                sha256_file(_metadata_path(args.shared_predictions))
                if args.shared_predictions
                else None
            )
        ),
        "shared_predictions_signature": (
            shared_metadata.get("signature") if shared_metadata else None
        ),
        "image_size": int((checkpoint.get("signature") or {}).get("image_size", 384)),
        "batch_size": int(args.batch_size),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "bootstrap_seed": int(args.seed),
        "diagnostic_limit": args.limit,
        "requested_device": str(args.device),
        "no_amp_requested": bool(args.no_amp),
        "amp_enabled": bool(amp_enabled),
        "runtime_environment": runtime_environment,
        "source_sha256": (
            dict(execution_source_identity["files"])
            if execution_source_identity is not None
            else legacy_source_identity
        ),
        "crop_policy": "highest-confidence detected xyxy; square 1.25 expansion",
        "reference_policy": "shared frozen production reference; detector fallback",
        "failure_nmae_penalty": 1.0,
    }


def validate_legacy_evaluation_authorization(
    verification_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Bind one frozen legacy VDN verification before deserialization."""

    verification_path = Path(verification_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    expected_root = Path(DEFAULT_LEGACY_RUN_ROOT).resolve()
    if checkpoint_path.name != "best.pt":
        raise ValueError("legacy VDN evaluation requires the formal best.pt")
    run_dir = checkpoint_path.parent
    try:
        seed = int(run_dir.name.removeprefix("seed_"))
    except ValueError as exc:
        raise ValueError("legacy VDN run directory has no formal seed") from exc
    expected_run_dir = expected_root / f"seed_{seed}"
    expected_verification = expected_run_dir / "verification.json"
    if (
        seed not in FORMAL_LEGACY_SEEDS
        or run_dir != expected_run_dir
        or verification_path != expected_verification
    ):
        raise ValueError("legacy VDN authorization is not a formal run artifact")
    summary_path = run_dir / "summary.json"
    last_path = run_dir / "last.pt"
    for path in (
        verification_path,
        checkpoint_path,
        summary_path,
        last_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    verification = _strict_json(verification_path)
    if set(verification) != _LEGACY_VERIFICATION_KEYS:
        raise ValueError("legacy VDN verification schema drifted")
    if (
        verification.get("protocol") != LEGACY_VERIFICATION_PROTOCOL
        or verification.get("verified") is not True
        or verification.get("run_dir") != str(run_dir)
        or int(verification.get("epochs", -1)) != 100
        or int(verification.get("group_overlap", -1)) != 0
    ):
        raise ValueError("legacy VDN verification is not eligible")
    expected_verifier_sha = sha256_source_file(
        PROJECT_DIR / "experiments" / "verify_vdn_run.py"
    )
    if verification.get("verifier_source_sha256") != expected_verifier_sha:
        raise ValueError("legacy VDN verifier source changed after verification")
    checkpoint_sha = sha256_file(checkpoint_path)
    if (
        verification.get("best_checkpoint_sha256") != checkpoint_sha
        or verification.get("last_checkpoint_sha256") != sha256_file(last_path)
        or verification.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ValueError("legacy VDN training artifact digest drifted")
    return {
        "protocol": LEGACY_VERIFICATION_PROTOCOL,
        "training_protocol": VDN_PROTOCOL,
        "parent_seed": seed,
        "run_dir": str(run_dir),
        "verification_path": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "verifier_source_sha256": expected_verifier_sha,
        "summary_sha256": verification["summary_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
    }


def _preauthorize_checkpoint(
    *,
    checkpoint_path: Path,
    phase2_cohort_authorization: Path | None,
    legacy_training_verification: Path | None,
) -> dict[str, Any]:
    supplied = sum(
        value is not None
        for value in (
            phase2_cohort_authorization,
            legacy_training_verification,
        )
    )
    if supplied != 1:
        raise PermissionError(
            "exactly one external authorization is required: "
            "--phase2-cohort-authorization or "
            "--legacy-training-verification"
        )
    if phase2_cohort_authorization is not None:
        authorization = validate_cohort_evaluation_authorization(
            phase2_cohort_authorization,
            checkpoint_path,
        )
        return {
            **authorization,
            "training_protocol": PHASE2_PROTOCOL,
        }
    assert legacy_training_verification is not None
    return validate_legacy_evaluation_authorization(
        legacy_training_verification,
        checkpoint_path,
    )


def _authorize_checkpoint_for_evaluation(
    checkpoint: dict[str, Any],
    *,
    external_authorization: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Cross-check safe checkpoint contents against an external authorization."""

    if not isinstance(checkpoint, dict):
        raise ValueError("VDN checkpoint is not an object")
    training_signature = checkpoint.get("signature")
    if not isinstance(training_signature, dict):
        raise ValueError("VDN checkpoint lacks a signed training signature")
    training_protocol = training_signature.get("protocol")
    expected_protocol = external_authorization.get("training_protocol")
    if expected_protocol not in (VDN_PROTOCOL, PHASE2_PROTOCOL):
        raise ValueError("external VDN authorization protocol is invalid")
    if training_protocol != expected_protocol:
        raise ValueError(
            "checkpoint self-reported protocol disagrees with external "
            "authorization"
        )
    if training_protocol not in (VDN_PROTOCOL, PHASE2_PROTOCOL):
        raise ValueError("checkpoint is not a signed SyncG-retrained VDN artifact")
    if (
        training_protocol == PHASE2_PROTOCOL
        and checkpoint.get("checkpoint_protocol") != PHASE2_CHECKPOINT_PROTOCOL
    ):
        raise ValueError("Phase-2 checkpoint protocol drifted")
    return str(training_protocol), dict(external_authorization)


def _resolve_evaluation_args(args: argparse.Namespace) -> None:
    args.phase2_public_preflight = getattr(
        args,
        "phase2_public_preflight",
        None,
    )
    args.manifest = args.manifest.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.vdn_source = args.vdn_source.resolve()
    args.output = args.output.resolve()
    args.meter_detector_weights = args.meter_detector_weights.resolve()
    args.keypoint_detector_weights = args.keypoint_detector_weights.resolve()
    if args.shared_predictions is not None:
        args.shared_predictions = args.shared_predictions.resolve()
    if args.phase2_cohort_authorization is not None:
        args.phase2_cohort_authorization = (
            args.phase2_cohort_authorization.resolve()
        )
    if args.phase2_public_preflight is not None:
        args.phase2_public_preflight = (
            args.phase2_public_preflight.resolve()
        )
    if args.legacy_training_verification is not None:
        args.legacy_training_verification = (
            args.legacy_training_verification.resolve()
        )


def _validate_output_does_not_alias_inputs(
    args: argparse.Namespace,
) -> None:
    output_paths = {
        args.output,
        _metadata_path(args.output),
        _summary_path(args.output),
        _journal_path(args.output),
        args.output.with_name(args.output.name + ".writer.lock"),
    }
    readonly_paths = {
        args.manifest,
        _protocol_path(args.manifest),
        args.checkpoint,
        args.meter_detector_weights,
        args.keypoint_detector_weights,
    }
    for optional in (
        args.shared_predictions,
        (
            _metadata_path(args.shared_predictions)
            if args.shared_predictions is not None
            else None
        ),
        args.phase2_cohort_authorization,
        getattr(args, "phase2_public_preflight", None),
        args.legacy_training_verification,
    ):
        if optional is not None:
            readonly_paths.add(Path(optional).resolve())
    collision = output_paths.intersection(readonly_paths)
    if collision:
        raise PermissionError(
            "evaluation output aliases a read-only input: "
            f"{sorted(str(path) for path in collision)}"
        )


def _prepare_evaluation(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Complete all read-only authorization before any output-side write."""

    _resolve_evaluation_args(args)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.batch_size <= 0 or args.bootstrap_iterations < 0:
        raise ValueError(
            "batch size must be positive and bootstrap iterations non-negative"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # The external report is validated before checkpoint deserialization and
    # before any test/public manifest, prediction cache, or image is opened.
    external_authorization = _preauthorize_checkpoint(
        checkpoint_path=args.checkpoint,
        phase2_cohort_authorization=args.phase2_cohort_authorization,
        legacy_training_verification=args.legacy_training_verification,
    )
    checkpoint = _safe_load_authorized_checkpoint(
        args.checkpoint,
        external_authorization,
    )
    training_protocol, training_authorization = (
        _authorize_checkpoint_for_evaluation(
            checkpoint,
            external_authorization=external_authorization,
        )
    )
    training_signature = checkpoint["signature"]
    if training_signature.get("vdn_source_commit") != verify_vdn_source(
        args.vdn_source
    ):
        raise ValueError("checkpoint and external VDN source commits differ")
    if (
        training_protocol == VDN_PROTOCOL
        and training_authorization.get("verifier_source_sha256")
        != sha256_source_file(
            PROJECT_DIR / "experiments" / "verify_vdn_run.py"
        )
    ):
        raise ValueError("legacy VDN external authorization changed")
    if training_protocol == PHASE2_PROTOCOL and args.overwrite:
        raise ValueError("formal Phase-2 evaluation forbids --overwrite")
    if (
        training_protocol == PHASE2_PROTOCOL
        and args.phase2_public_preflight is None
    ):
        raise PermissionError(
            "formal Phase-2 evaluation requires the public preflight "
            "authorization"
        )
    if (
        training_protocol != PHASE2_PROTOCOL
        and args.phase2_public_preflight is not None
    ):
        raise ValueError(
            "public preflight authorization is only valid for Phase-2"
        )

    phase2_plan: dict[str, Any] | None = None
    phase2_public_preflight: dict[str, Any] | None = None
    authorized_input_bytes: dict[str, bytes] | None = None
    if training_protocol == PHASE2_PROTOCOL:
        _validate_phase2_process_isolation()
        phase2_plan = validate_phase2_evaluation_plan(
            args,
            args.checkpoint,
        )
        if (
            training_authorization.get("evaluation_plan_source_sha256")
            != phase2_plan["plan_source_sha256"]
        ):
            raise ValueError(
                "cohort authorization and evaluation plan source differ"
            )
        if (
            training_authorization.get(
                "evaluation_execution_source_identity"
            )
            != phase2_plan["execution_source_identity"]
        ):
            raise ValueError(
                "cohort authorization and evaluation execution sources differ"
            )
        assert args.phase2_cohort_authorization is not None
        assert args.phase2_public_preflight is not None
        phase2_public_preflight = validate_phase2_public_preflight(
            args.phase2_public_preflight,
            args.phase2_cohort_authorization,
            expected_plan=phase2_plan,
        )
        authorized_input_bytes = _snapshot_phase2_inputs(phase2_plan)
    _validate_output_does_not_alias_inputs(args)
    return {
        "checkpoint": checkpoint,
        "training_protocol": training_protocol,
        "training_authorization": training_authorization,
        "phase2_evaluation_plan": phase2_plan,
        "phase2_public_preflight": phase2_public_preflight,
        "authorized_input_bytes": authorized_input_bytes,
    }


def _run_evaluation(
    args: argparse.Namespace,
    *,
    prepared: dict[str, Any] | None = None,
) -> None:
    if prepared is None:
        prepared = _prepare_evaluation(args)
    checkpoint = prepared["checkpoint"]
    training_protocol = str(prepared["training_protocol"])
    training_authorization = dict(prepared["training_authorization"])
    phase2_evaluation_plan = prepared.get("phase2_evaluation_plan")
    phase2_public_preflight = prepared.get("phase2_public_preflight")
    authorized_input_bytes = prepared.get("authorized_input_bytes")
    training_signature = checkpoint["signature"]

    _configure_evaluation_determinism(training_protocol)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp
    runtime_environment = _runtime_environment(device)

    required = [
        args.manifest,
        args.meter_detector_weights,
        args.keypoint_detector_weights,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required files are missing: {missing}")

    rows = (
        _read_jsonl(args.manifest)
        if authorized_input_bytes is None
        else _read_jsonl_bytes(
            authorized_input_bytes["manifest"],
            label=str(args.manifest),
        )
    )
    if args.limit is not None:
        rows = rows[: max(0, int(args.limit))]
    if not rows:
        raise ValueError("selected manifest is empty")
    public_release_inventory_sha256: str | None = None
    authorized_image_sha256: dict[str, str] = {}
    if phase2_evaluation_plan is not None:
        public_scope = (
            "rpm10k_support"
            if phase2_evaluation_plan["name"] == "rpm10k"
            else "syncg_test"
        )
        (
            public_release_inventory_sha256,
            authorized_image_sha256,
        ) = _authorize_public_image_inventory(
            rows,
            public_scope=public_scope,
        )
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            row["_authorized_input_image_sha256"] = (
                authorized_image_sha256[sample_id]
            )
    shared, shared_metadata = _load_shared_predictions(
        args.shared_predictions,
        manifest=args.manifest,
        rows=rows,
        condition=args.condition,
        degradation_seed=args.degradation_seed,
        meter_weights=args.meter_detector_weights,
        point_weights=args.keypoint_detector_weights,
        authorized_inputs=authorized_input_bytes,
        plan_identity=phase2_evaluation_plan,
    )

    image_size = int(training_signature.get("image_size", 384))
    signature = _run_signature(
        args,
        checkpoint,
        shared_metadata,
        training_protocol=training_protocol,
        training_authorization=training_authorization,
        phase2_evaluation_plan=phase2_evaluation_plan,
        phase2_public_preflight=phase2_public_preflight,
        public_release_inventory_sha256=(
            public_release_inventory_sha256
        ),
        runtime_environment=runtime_environment,
        amp_enabled=amp_enabled,
    )
    metadata_path = _metadata_path(args.output)
    summary_path = _summary_path(args.output)
    journal_path = _journal_path(args.output)
    expected_ids = [str(row.get("sample_id") or "") for row in rows]
    if (
        any(not sample_id for sample_id in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise ValueError("evaluation manifest has empty or duplicate sample IDs")
    expected_id_set = set(expected_ids)

    if args.overwrite:
        args.output.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        journal_path.unlink(missing_ok=True)

    output_exists = args.output.is_file()
    metadata_exists = metadata_path.is_file()
    summary_exists = summary_path.is_file()
    journal_exists = journal_path.is_file()
    completed: set[str] = set()

    if summary_exists:
        if not output_exists or not metadata_exists or not journal_exists:
            raise RuntimeError(
                "completed summary is orphaned from predictions, metadata, "
                "or append journal"
            )
        if not args.resume:
            raise FileExistsError(
                f"immutable evaluation already exists: {summary_path}"
            )
        previous = _strict_json(metadata_path)
        if previous.get("signature") != signature:
            raise ValueError("VDN evaluation resume signature mismatch")
        _PredictionJournal(
            args.output,
            journal_path,
            resume=True,
        )
        existing_rows = _read_jsonl(args.output)
        existing_ids = [
            str(row.get("sample_id") or "") for row in existing_rows
        ]
        if (
            any(not sample_id for sample_id in existing_ids)
            or len(existing_ids) != len(set(existing_ids))
            or set(existing_ids) != expected_id_set
        ):
            raise ValueError("completed VDN predictions are incomplete or duplicated")
        existing_summary = _strict_json(summary_path)
        if (
            existing_summary.get("status") != "complete"
            or existing_summary.get("protocol") != signature["protocol"]
            or existing_summary.get("signature") != signature
            or existing_summary.get("predictions_sha256")
            != sha256_file(args.output)
            or existing_summary.get("metadata_sha256")
            != sha256_file(metadata_path)
            or existing_summary.get("journal_sha256")
            != sha256_file(journal_path)
        ):
            raise ValueError("completed VDN evaluation artifact drifted")
        print(
            json.dumps(
                existing_summary["metrics"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        print(summary_path)
        return

    partial_artifacts = sum(
        (output_exists, metadata_exists, journal_exists)
    )
    if partial_artifacts not in (0, 3):
        raise RuntimeError(
            "partial evaluation must contain predictions, metadata, and "
            "append journal"
        )
    if output_exists:
        if not args.resume:
            raise FileExistsError(
                f"incomplete evaluation exists; pass --resume: {args.output}"
            )
        previous = _strict_json(metadata_path)
        if previous.get("signature") != signature:
            raise ValueError("VDN evaluation resume signature mismatch")
        prediction_journal = _PredictionJournal(
            args.output,
            journal_path,
            resume=True,
            recover_uncommitted_tail=True,
        )
        existing_rows = _read_jsonl(args.output)
        existing_ids = [
            str(row.get("sample_id") or "") for row in existing_rows
        ]
        if (
            any(not sample_id for sample_id in existing_ids)
            or len(existing_ids) != len(set(existing_ids))
            or not set(existing_ids).issubset(expected_id_set)
        ):
            raise ValueError(
                "resume predictions contain duplicate, empty, or unknown sample IDs"
            )
        completed = set(existing_ids)
    else:
        if args.resume:
            raise FileNotFoundError(
                "resume requested but no incomplete prediction/metadata pair exists"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8"):
            pass
        with journal_path.open("x", encoding="utf-8"):
            pass
        metadata = {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "shared_predictions": (
                str(args.shared_predictions) if args.shared_predictions else None
            ),
            "signature": signature,
            "environment": runtime_environment,
            "append_journal": str(journal_path),
        }
        _atomic_json_no_clobber(metadata_path, metadata)
        prediction_journal = _PredictionJournal(
            args.output,
            journal_path,
            resume=False,
        )

    pending = [row for row in rows if str(row.get("sample_id")) not in completed]
    model = build_vdn_model(
        args.vdn_source,
        image_size=image_size,
        imagenet_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    target_detector_class = _load_target_detector_class()
    if authorized_input_bytes is None:
        meter_detector = target_detector_class(
            str(args.meter_detector_weights)
        )
        point_detector: Any | None = None
    else:
        with tempfile.TemporaryDirectory(
            prefix="vdn-authorized-weights-"
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            meter_snapshot = temporary_root / "meter.pt"
            point_snapshot = temporary_root / "point.pt"
            meter_snapshot.write_bytes(
                authorized_input_bytes["meter_detector_weights"]
            )
            point_snapshot.write_bytes(
                authorized_input_bytes["keypoint_detector_weights"]
            )
            meter_detector = target_detector_class(str(meter_snapshot))
            point_detector = target_detector_class(str(point_snapshot))
    batch: list[dict[str, Any]] = []

    @torch.inference_mode()
    def flush_batch() -> None:
        if not batch:
            return
        inputs = torch.stack([item["tensor"] for item in batch]).to(
            device,
            non_blocking=True,
        )
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            heatmaps, vector_maps = model(inputs)
        directions, peaks, valid = predict_directions(
            heatmaps.float(),
            vector_maps.float(),
        )
        directions_np = directions.detach().cpu().numpy()
        peaks_np = peaks.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        output_rows: list[dict[str, Any]] = []
        for index, item in enumerate(batch):
            row = item["row"]
            degradation = item["degradation"]
            runtime = time.perf_counter() - item["started"]
            if not bool(valid_np[index]):
                output_rows.append(
                    _failure_result(
                        row,
                        degradation,
                        code="invalid_direction",
                        message="VDN vector at the predicted tip has zero/invalid norm",
                        runtime_seconds=runtime,
                    )
                )
                continue
            direction = directions_np[index].astype(np.float64)
            try:
                pointer_angle = image_angle_from_direction(direction)
                reading, progress = reading_from_pointer_angle(
                    pointer_angle,
                    start_angle=item["start_angle"],
                    range_angle=item["range_angle"],
                    scale_start=float(row["scale_start"]),
                    scale_end=float(row["scale_end"]),
                )
            except ValueError as exc:
                output_rows.append(
                    _failure_result(
                        row,
                        degradation,
                        code="reading_conversion_failed",
                        message=str(exc),
                        runtime_seconds=runtime,
                    )
                )
                continue
            result = _base_result(row, degradation)
            result.update(
                {
                    "status": True,
                    "prediction": float(reading),
                    "progress": float(progress),
                    "pointer_angle": float(pointer_angle),
                    "direction": direction.tolist(),
                    "direction_angle_error_degrees": _direction_error(
                        direction,
                        item["target_direction"],
                    ),
                    "heatmap_peak": float(peaks_np[index]),
                    "meter_bbox": list(item["bbox"]),
                    "meter_confidence": item["meter_confidence"],
                    "start_angle": float(item["start_angle"]),
                    "range_angle": float(item["range_angle"]),
                    "reference_branch": item["reference_branch"],
                    "reference_source": item["reference_source"],
                    "runtime_seconds": float(runtime),
                }
            )
            output_rows.append(result)
        prediction_journal.append(output_rows)
        batch.clear()

    for row in tqdm(pending, desc=f"VDN {args.condition}", dynamic_ncols=True):
        started = time.perf_counter()
        sample_id = str(row.get("sample_id"))
        degradation: dict[str, Any] = {
            "protocol": ROBUSTNESS_PROTOCOL,
            "condition": args.condition,
            "seed": int(args.degradation_seed),
        }
        image_value = row.get("image_path")
        image_path = Path(str(image_value))
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path
        if authorized_image_sha256:
            image_payload = image_path.read_bytes()
            actual_image_sha256 = hashlib.sha256(
                image_payload
            ).hexdigest()
            expected_image_sha256 = authorized_image_sha256[sample_id]
            if actual_image_sha256 != expected_image_sha256:
                raise RuntimeError(
                    "public input image changed after release authorization: "
                    f"{sample_id}"
                )
            image = cv2.imdecode(
                np.frombuffer(image_payload, dtype=np.uint8),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
        else:
            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
        if image is None:
            prediction_journal.append(
                [
                    _failure_result(
                        row,
                        degradation,
                        code="image_read_failed",
                        message=f"failed to read {image_path}",
                        runtime_seconds=time.perf_counter() - started,
                    )
                ],
            )
            continue
        try:
            image, degradation = apply_degradation(
                image,
                args.condition,
                sample_id=sample_id,
                seed=args.degradation_seed,
            )
            confidences, boxes, crops, _, best_index = meter_detector.image_crop(image)
            if best_index is None or not crops:
                prediction_journal.append(
                    [
                        _failure_result(
                            row,
                            degradation,
                            code="meter_not_found",
                            message="shared meter detector found no dial",
                            runtime_seconds=time.perf_counter() - started,
                        )
                    ],
                )
                continue
            crop = crops[best_index]
            bbox = _xyxy_from_detector_box(boxes[best_index])
            reference = _shared_reference(shared.get(sample_id))
            if reference is None:
                if point_detector is None:
                    point_detector = target_detector_class(
                        str(args.keypoint_detector_weights)
                    )
                reference = _fallback_reference(point_detector, crop)
                reference_source = "fallback_keypoint_detector"
            else:
                reference_source = "shared_frozen_production_cache"
            start_angle, range_angle, reference_branch = reference
            tensor = vdn_tensor_from_bbox(
                image,
                bbox,
                image_size=image_size,
            )
            batch.append(
                {
                    "row": row,
                    "degradation": degradation,
                    "tensor": tensor,
                    "bbox": bbox,
                    "meter_confidence": float(confidences[best_index]),
                    "start_angle": start_angle,
                    "range_angle": range_angle,
                    "reference_branch": reference_branch,
                    "reference_source": reference_source,
                    "target_direction": _target_direction(row, degradation),
                    "started": started,
                }
            )
            if len(batch) >= args.batch_size:
                flush_batch()
        except Exception as exc:
            prediction_journal.append(
                [
                    _failure_result(
                        row,
                        degradation,
                        code="pipeline_exception",
                        message=f"{type(exc).__name__}: {exc}",
                        runtime_seconds=time.perf_counter() - started,
                    )
                ],
            )
    flush_batch()

    output_rows = _read_jsonl(args.output)
    output_ids = [str(row.get("sample_id")) for row in output_rows]
    if (
        len(output_ids) != len(set(output_ids))
        or set(output_ids) != expected_id_set
    ):
        raise RuntimeError("VDN output is incomplete or contains duplicate sample IDs")
    summary = {
        "schema_version": 2,
        "protocol": signature["protocol"],
        "status": "complete",
        "condition": args.condition,
        "signature": signature,
        "predictions_sha256": sha256_file(args.output),
        "metadata_sha256": sha256_file(metadata_path),
        "journal_sha256": sha256_file(journal_path),
        "metrics": summarize_scalar_predictions(
            output_rows,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "dialbench_metrics": _dialbench_summary(output_rows),
        "direction_component": _component_summary(output_rows),
        "subgroups": _subgroup_summaries(output_rows),
        "failure_codes": {
            code: sum(row.get("error_code") == code for row in output_rows)
            for code in sorted(
                {str(row.get("error_code")) for row in output_rows if row.get("error_code")}
            )
        },
    }
    _atomic_json_no_clobber(summary_path, summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, sort_keys=True))
    print(summary_path)


def main() -> None:
    args = parse_args()
    prepared = _prepare_evaluation(args)
    with _exclusive_output_lock(args.output):
        _run_evaluation(args, prepared=prepared)


if __name__ == "__main__":
    main()
