"""Frozen, CPU-only adapters for mature off-the-shelf numeric OCR.

The adapters in this module never download models.  Every runtime and model
artifact must first be captured in a model-bundle JSON and is re-authenticated
before inference.  PP-OCRv5 is the primary path; the already cached
PP-OCRv4/RapidOCR runtime is retained as a zero-download control.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range import (
    OCRToken,
    sha256_file,
    validate_canonical_roi,
)
from experiments.automatic_numeric_range_public_protocol import (
    canonical_sha256,
    require,
    strict_json,
)
from experiments.syncg_numeric_ocr import (
    CTCStringHypothesis,
    OCRPosteriorToken,
    normalize_numeric_text,
)


BUNDLE_PROTOCOL: Final[str] = "pretrained_numeric_ocr_model_bundle_v1"
ADAPTER_PROTOCOL: Final[str] = "pretrained_numeric_ocr_adapter_v1"
SAFE_ROOT: Final[Path] = Path(r"C:\pointer_read").resolve()
SUPPORTED_BACKENDS: Final[frozenset[str]] = frozenset(
    {"rapidocr_ppocrv4_mobile", "paddleocr_ppocrv5"}
)


def _safe_existing(path: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(SAFE_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay below {SAFE_ROOT}") from error
    return resolved


def _safe_new_file(path: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(SAFE_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay below {SAFE_ROOT}") from error
    require(resolved != SAFE_ROOT, f"refusing broad {label}")
    require(not resolved.exists(), f"refusing to overwrite {label}: {resolved}")
    return resolved


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = _safe_new_file(path, label="model bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                dict(value), ensure_ascii=False, indent=2, sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def directory_identity(path: Path) -> dict[str, Any]:
    """Return a stable content identity without storing a huge file table."""

    root = _safe_existing(path, label="runtime/model directory")
    require(root.is_dir(), f"expected directory: {root}")
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for file in sorted((item for item in root.rglob("*") if item.is_file())):
        relative = file.relative_to(root).as_posix()
        size = int(file.stat().st_size)
        rows.append(
            {"relative_path": relative, "bytes": size, "sha256": sha256_file(file)}
        )
        total_bytes += size
    require(bool(rows), f"empty directory: {root}")
    return {
        "path": str(root),
        "files": len(rows),
        "bytes": total_bytes,
        "inventory_sha256": canonical_sha256(rows),
    }


def file_identity(path: Path, *, label: str) -> dict[str, Any]:
    file = _safe_existing(path, label=label)
    require(file.is_file(), f"expected file: {file}")
    return {
        "path": str(file),
        "bytes": int(file.stat().st_size),
        "sha256": sha256_file(file),
    }


def _verify_identity(value: Mapping[str, Any], *, label: str) -> Path:
    path = _safe_existing(Path(str(value.get("path") or "")), label=label)
    if path.is_file():
        observed = file_identity(path, label=label)
        for key in ("bytes", "sha256"):
            require(observed[key] == value.get(key), f"{label} {key} drift")
    else:
        observed = directory_identity(path)
        for key in ("files", "bytes", "inventory_sha256"):
            require(observed[key] == value.get(key), f"{label} {key} drift")
    return path


def _runtime_distributions(runtime_root: Path, names: Sequence[str]) -> dict[str, str]:
    """Read versions from a target directory without importing its packages."""

    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions(path=[str(runtime_root)]):
        name = str(distribution.metadata.get("Name") or "").casefold()
        if name in {item.casefold() for item in names}:
            result[name] = str(distribution.version)
    return dict(sorted(result.items()))


def freeze_model_bundle(
    *,
    candidate_id: str,
    backend_kind: str,
    runtime_root: Path,
    artifacts: Mapping[str, Path],
    output_path: Path,
    model_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze already-present runtime/model files before any evaluation."""

    require(backend_kind in SUPPORTED_BACKENDS, "unsupported pretrained OCR backend")
    runtime = directory_identity(runtime_root)
    marker = (
        Path(runtime["path"]) / "rapidocr/__init__.py"
        if backend_kind == "rapidocr_ppocrv4_mobile"
        else Path(runtime["path"]) / "paddleocr/__init__.py"
    )
    require(marker.is_file(), f"runtime package marker absent: {marker}")
    required = (
        {"detector", "recognizer", "dictionary"}
        if backend_kind == "rapidocr_ppocrv4_mobile"
        else {"recognizer_model"}
    )
    require(required <= set(artifacts), f"missing artifacts: {sorted(required - set(artifacts))}")
    frozen_artifacts: dict[str, Any] = {}
    for name, path in sorted(artifacts.items()):
        resolved = _safe_existing(Path(path), label=f"{candidate_id}.{name}")
        if backend_kind == "paddleocr_ppocrv5" and name.endswith("_model"):
            require(resolved.is_dir(), f"{candidate_id}.{name} must be an extracted model directory")
            require(
                (resolved / "inference.pdiparams").is_file(),
                f"{candidate_id}.{name} lacks inference.pdiparams",
            )
            require(
                (resolved / "inference.json").is_file()
                or (resolved / "inference.pdmodel").is_file(),
                f"{candidate_id}.{name} lacks a Paddle inference graph",
            )
        frozen_artifacts[name] = (
            file_identity(resolved, label=f"{candidate_id}.{name}")
            if resolved.is_file()
            else directory_identity(resolved)
        )
    distributions = _runtime_distributions(
        Path(runtime["path"]),
        (
            ("rapidocr", "onnxruntime", "numpy")
            if backend_kind == "rapidocr_ppocrv4_mobile"
            else ("paddleocr", "paddlex", "paddlepaddle", "onnxruntime")
        ),
    )
    value = {
        "schema_version": 1,
        "protocol": BUNDLE_PROTOCOL,
        "status": "frozen_before_public_evaluation",
        "candidate_id": str(candidate_id),
        "backend_kind": backend_kind,
        "execution_device": "cpu",
        "implicit_download_allowed": False,
        "runtime": runtime,
        "runtime_distributions": distributions,
        "artifacts": frozen_artifacts,
        "model_names": dict(model_names or {}),
        "adapter_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    _atomic_json(output_path, value)
    return value


def load_model_bundle(path: Path) -> tuple[Path, dict[str, Any]]:
    bundle_path = _safe_existing(path, label="pretrained OCR model bundle")
    require(bundle_path.is_file(), "model bundle must be a JSON file")
    value = strict_json(bundle_path)
    require(value.get("protocol") == BUNDLE_PROTOCOL, "model bundle protocol drift")
    require(value.get("status") == "frozen_before_public_evaluation", "model bundle not frozen")
    require(value.get("backend_kind") in SUPPORTED_BACKENDS, "model bundle backend drift")
    require(value.get("execution_device") == "cpu", "pretrained gate must remain CPU-only")
    require(value.get("implicit_download_allowed") is False, "implicit downloads must be disabled")
    _verify_identity(value["runtime"], label="OCR runtime")
    for name, identity in value.get("artifacts", {}).items():
        require(isinstance(identity, Mapping), f"bad artifact identity: {name}")
        _verify_identity(identity, label=f"OCR artifact {name}")
    source = value.get("adapter_source")
    require(isinstance(source, Mapping), "adapter source binding missing")
    require(
        Path(str(source.get("path") or "")).resolve(strict=True)
        == Path(__file__).resolve(strict=True),
        "adapter source path drift",
    )
    require(source.get("sha256") == sha256_file(Path(__file__)), "adapter source hash drift")
    return bundle_path, value


def normalized_prediction_text(value: Any) -> tuple[str, bool]:
    """Apply exactly the same conservative numeric punctuation policy as Tiny."""

    raw = str(value).strip().translate(
        str.maketrans({"−": "-", "–": "-", "—": "-", "﹣": "-", "－": "-"})
    )
    if "," in raw and "." not in raw and raw.count(",") == 1:
        raw = raw.replace(",", ".")
    try:
        return normalize_numeric_text(raw), True
    except ValueError:
        return raw, False


def _result_mapping(value: Any) -> Mapping[str, Any]:
    """Normalize PaddleOCR result objects across 3.x minor releases."""

    if isinstance(value, Mapping):
        result: Any = value
    else:
        result = None
        for name in ("json", "res", "to_dict"):
            candidate = getattr(value, name, None)
            if candidate is None:
                continue
            candidate = candidate() if callable(candidate) else candidate
            if isinstance(candidate, Mapping):
                result = candidate
                break
        if result is None:
            raise TypeError("unsupported PaddleOCR result object")
    if isinstance(result.get("res"), Mapping):
        result = result["res"]
    return result


def parse_paddle_recognition_result(value: Any) -> tuple[str, float]:
    result = _result_mapping(value)
    text = result.get("rec_text")
    score = result.get("rec_score")
    require(text is not None and score is not None, "Paddle recognition result missing fields")
    confidence = float(score)
    require(math.isfinite(confidence), "Paddle recognition score is non-finite")
    return str(text), float(np.clip(confidence, 0.0, 1.0))


def parse_paddle_pipeline_result(
    value: Any,
) -> tuple[list[tuple[tuple[tuple[float, float], ...], str, float, float]], Mapping[str, Any]]:
    result = _result_mapping(value)
    boxes = result.get("dt_polys")
    texts = result.get("rec_texts")
    scores = result.get("rec_scores")
    detector_scores = result.get("dt_scores")
    if boxes is None or texts is None or scores is None:
        return [], result
    require(len(boxes) == len(texts) == len(scores), "Paddle pipeline output length drift")
    if detector_scores is not None:
        require(len(detector_scores) == len(boxes), "Paddle detector score length drift")
    rows = []
    for index, (box, text, score) in enumerate(zip(boxes, texts, scores, strict=True)):
        points = np.asarray(box, dtype=np.float64)
        require(points.shape == (4, 2) and np.isfinite(points).all(), "invalid Paddle text box")
        rec_score = float(np.clip(float(score), 0.0, 1.0))
        det_score = (
            rec_score
            if detector_scores is None
            else float(np.clip(float(detector_scores[index]), 0.0, 1.0))
        )
        rows.append(
            (
                tuple((float(x), float(y)) for x, y in points),
                str(text),
                rec_score,
                det_score,
            )
        )
    return rows, result


@runtime_checkable
class MatureOCRAdapter(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def recognize(self, crops_bgr: Sequence[np.ndarray]) -> list[tuple[str, float]]: ...

    def infer_with_posteriors(
        self, image_bgr: np.ndarray
    ) -> tuple[list[OCRToken], list[OCRPosteriorToken], float]: ...


@dataclass
class RapidOCRV4Adapter:
    bundle_path: Path
    bundle: Mapping[str, Any]
    cpu_threads: int = 2

    def __post_init__(self) -> None:
        runtime = Path(self.bundle["runtime"]["path"])
        # A frozen target runtime must not mutate itself by creating new pyc
        # files after its content inventory has been authenticated.
        sys.dont_write_bytecode = True
        if str(runtime) not in sys.path:
            sys.path.append(str(runtime))
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        artifacts = self.bundle["artifacts"]
        params = {
            "Global.log_level": "error",
            "Global.use_cls": False,
            "Global.text_score": 0.0,
            "Global.max_side_len": 2000,
            "Det.model_path": artifacts["detector"]["path"],
            "Rec.model_path": artifacts["recognizer"]["path"],
            "Rec.rec_keys_path": artifacts["dictionary"]["path"],
            "EngineConfig.onnxruntime.intra_op_num_threads": int(self.cpu_threads),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
        }
        self._engine = RapidOCR(params=params)
        self._identity = {
            "protocol": ADAPTER_PROTOCOL,
            "candidate_id": self.bundle["candidate_id"],
            "backend_kind": self.bundle["backend_kind"],
            "bundle": {
                "path": str(self.bundle_path),
                "sha256": sha256_file(self.bundle_path),
            },
            "execution_device": "cpu",
            "implicit_downloads": False,
            "posterior_semantics": "single_top1_hypothesis",
            "parameters": params,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def recognize(self, crops_bgr: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for crop in crops_bgr:
            output = self._engine(
                np.ascontiguousarray(crop), use_det=False, use_cls=False, use_rec=True
            )
            texts = getattr(output, "txts", None)
            scores = getattr(output, "scores", None)
            if not texts or not scores:
                results.append(("", 0.0))
            else:
                results.append((str(texts[0]), float(scores[0])))
        return results

    def infer_with_posteriors(
        self, image_bgr: np.ndarray
    ) -> tuple[list[OCRToken], list[OCRPosteriorToken], float]:
        image = validate_canonical_roi(image_bgr)
        started = time.perf_counter()
        output = self._engine(image, use_det=True, use_cls=False, use_rec=True)
        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        if boxes is None or texts is None or scores is None:
            return [], [], time.perf_counter() - started
        rows = [
            (
                tuple((float(x), float(y)) for x, y in np.asarray(box)),
                str(text), float(score), float(score),
            )
            for box, text, score in zip(boxes, texts, scores, strict=True)
        ]
        return _posterior_outputs(rows, time.perf_counter() - started)


@dataclass
class PaddleOCRV5Adapter:
    bundle_path: Path
    bundle: Mapping[str, Any]
    recognition_batch_size: int = 32

    def __post_init__(self) -> None:
        runtime = Path(self.bundle["runtime"]["path"])
        sys.dont_write_bytecode = True
        if str(runtime) not in sys.path:
            sys.path.append(str(runtime))
        from paddleocr import PaddleOCR, TextRecognition  # type: ignore[import-not-found]

        artifacts = self.bundle["artifacts"]
        names = self.bundle.get("model_names", {})
        recognition_name = str(names.get("recognizer") or "")
        require(recognition_name.startswith("PP-OCRv5") or recognition_name.startswith("en_PP-OCRv5"), "unexpected PP-OCRv5 recognizer name")
        recognition_root = str(Path(artifacts["recognizer_model"]["path"]))
        self._recognizer = TextRecognition(
            model_name=recognition_name,
            model_dir=recognition_root,
            device="cpu",
            enable_hpi=False,
        )
        self._pipeline = None
        if "detector_model" in artifacts:
            detection_name = str(names.get("detector") or "")
            require(detection_name.startswith("PP-OCRv5"), "unexpected PP-OCRv5 detector name")
            self._pipeline = PaddleOCR(
                text_detection_model_name=detection_name,
                text_detection_model_dir=str(Path(artifacts["detector_model"]["path"])),
                text_recognition_model_name=recognition_name,
                text_recognition_model_dir=recognition_root,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_rec_score_thresh=0.0,
                device="cpu",
            )
        self._identity = {
            "protocol": ADAPTER_PROTOCOL,
            "candidate_id": self.bundle["candidate_id"],
            "backend_kind": self.bundle["backend_kind"],
            "bundle": {
                "path": str(self.bundle_path),
                "sha256": sha256_file(self.bundle_path),
            },
            "execution_device": "cpu",
            "implicit_downloads": False,
            "model_names": dict(names),
            "posterior_semantics": "single_top1_hypothesis",
            "recognition_batch_size": int(self.recognition_batch_size),
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def recognize(self, crops_bgr: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        if not crops_bgr:
            return []
        output = self._recognizer.predict(
            input=[np.ascontiguousarray(crop) for crop in crops_bgr],
            batch_size=max(1, min(int(self.recognition_batch_size), len(crops_bgr))),
        )
        rows = list(output)
        require(len(rows) == len(crops_bgr), "Paddle recognizer output length drift")
        return [parse_paddle_recognition_result(row) for row in rows]

    def infer_with_posteriors(
        self, image_bgr: np.ndarray
    ) -> tuple[list[OCRToken], list[OCRPosteriorToken], float]:
        require(self._pipeline is not None, "PP-OCRv5 detector model is absent")
        image = validate_canonical_roi(image_bgr)
        started = time.perf_counter()
        output = list(self._pipeline.predict(input=image))
        elapsed = time.perf_counter() - started
        if not output:
            return [], [], elapsed
        require(len(output) == 1, "Paddle OCR emitted multiple pages for one image")
        rows, _ = parse_paddle_pipeline_result(output[0])
        return _posterior_outputs(rows, elapsed)


def _posterior_outputs(
    rows: Sequence[tuple[tuple[tuple[float, float], ...], str, float, float]],
    elapsed: float,
) -> tuple[list[OCRToken], list[OCRPosteriorToken], float]:
    tokens: list[OCRToken] = []
    posteriors: list[OCRPosteriorToken] = []
    for box, raw_text, recognition_score, detector_score in rows:
        normalized, parseable = normalized_prediction_text(raw_text)
        hypotheses: tuple[CTCStringHypothesis, ...] = ()
        if parseable:
            probability = float(np.clip(recognition_score, 1e-8, 1.0))
            hypotheses = (
                CTCStringHypothesis(
                    text=normalized,
                    log_probability=float(math.log(probability)),
                    beam_probability=probability,
                ),
            )
            tokens.append(
                OCRToken(
                    text=normalized,
                    score=float(math.sqrt(max(0.0, detector_score * recognition_score))),
                    box=box,
                ).validate()
            )
        posteriors.append(
            OCRPosteriorToken(
                box=box,
                detector_score=float(np.clip(detector_score, 0.0, 1.0)),
                hypotheses=hypotheses,
            )
        )
    require(math.isfinite(float(elapsed)) and elapsed >= 0.0, "invalid OCR runtime")
    return tokens, posteriors, float(elapsed)


def load_adapter(bundle_path: Path, *, cpu_threads: int = 2) -> MatureOCRAdapter:
    resolved, bundle = load_model_bundle(bundle_path)
    if bundle["backend_kind"] == "rapidocr_ppocrv4_mobile":
        return RapidOCRV4Adapter(resolved, bundle, cpu_threads=max(1, int(cpu_threads)))
    return PaddleOCRV5Adapter(resolved, bundle)


__all__ = [
    "ADAPTER_PROTOCOL",
    "BUNDLE_PROTOCOL",
    "MatureOCRAdapter",
    "PaddleOCRV5Adapter",
    "RapidOCRV4Adapter",
    "directory_identity",
    "file_identity",
    "freeze_model_bundle",
    "load_adapter",
    "load_model_bundle",
    "normalized_prediction_text",
    "parse_paddle_pipeline_result",
    "parse_paddle_recognition_result",
]
