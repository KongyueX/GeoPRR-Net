"""Label-free automatic numeric range inference for a canonical meter ROI.

The module deliberately separates three concerns:

* a geometry provider predicts a pivot and ordered ScaleMark endpoints;
* an OCR backend observes text in the *same whole* canonical ROI; and
* :func:`decode_numeric_range` fits a robust arithmetic scale from OCR box
  centres expressed as progress along the predicted ScaleMark arc.

Ground-truth text boxes, text strings, physical scale values, meter boxes and
manual ScaleMarks are not accepted by the primary inference API.  SyncG's
public annotations may supervise a future detector/CTC backend, but they are
never consumed here at inference time.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np


PROTOCOL: Final[str] = "automatic_numeric_range_geometry_guided_ocr_v1"
PREDICTION_SPACE: Final[str] = "real_numeric_scale_start_end"
DEFAULT_OCR_RUNTIME = Path(r"C:\pointer_read\rapidocr_runtime")

_NUMERIC_RE = re.compile(r"^[+-]?(?:\d{1,6}(?:\.\d{1,4})?|\.\d{1,4})$")
_MINUS_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
        "－": "-",
        "＋": "+",
    }
)
_FORBIDDEN_GEOMETRY_KEYS = frozenset(
    {
        "actual",
        "groundtruth",
        "geometry",
        "gt",
        "label",
        "labels",
        "manualreference",
        "manualscalemark",
        "meterbbox",
        "physicalscaleend",
        "physicalscalestart",
        "reference",
        "referencepacket",
        "scaleend",
        "scalestart",
        "scalemark",
        "target",
        "textbboxannotations",
        "truth",
    }
)


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def reject_supervised_fields(value: Any, *, path: str = "input") -> None:
    """Recursively reject label/manual fields before any image processing."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_key(key) in _FORBIDDEN_GEOMETRY_KEYS:
                raise ValueError(f"forbidden supervised/manual field at {path}.{key}")
            reject_supervised_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_supervised_fields(nested, path=f"{path}[{index}]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_sha256(image_bgr: np.ndarray) -> str:
    image = validate_canonical_roi(image_bgr)
    digest = hashlib.sha256()
    digest.update(str(tuple(image.shape)).encode("ascii"))
    digest.update(b"\0BGR_UINT8\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def validate_canonical_roi(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("canonical_meter_roi_bgr must be a NumPy array")
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("canonical meter ROI must have shape [H,W,3] and dtype uint8")
    if min(image_bgr.shape[:2]) < 32:
        raise ValueError("canonical meter ROI is too small")
    return np.ascontiguousarray(image_bgr)


@dataclass(frozen=True)
class GeometryHint:
    """Automatic, normalized geometry for one whole canonical ROI."""

    pivot_xy: tuple[float, float]
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    confidence: float
    source: str

    def validate(self) -> "GeometryHint":
        points = (self.pivot_xy, self.start_xy, self.end_xy)
        for name, point in zip(("pivot", "start", "end"), points, strict=True):
            if len(point) != 2 or not np.isfinite(point).all():
                raise ValueError(f"{name}_xy must contain two finite values")
            if not all(-0.05 <= float(value) <= 1.05 for value in point):
                raise ValueError(f"{name}_xy lies outside the normalized ROI")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("geometry confidence must be in [0,1]")
        if not str(self.source).strip():
            raise ValueError("geometry source must be non-empty")
        start_radius = math.dist(self.pivot_xy, self.start_xy)
        end_radius = math.dist(self.pivot_xy, self.end_xy)
        if min(start_radius, end_radius) <= 0.02:
            raise ValueError("ScaleMark endpoints collapse onto the pivot")
        arc = directed_arc_radians(self)
        if not math.radians(10.0) < arc < math.radians(350.0):
            raise ValueError("predicted ScaleMark arc is outside (10,350) degrees")
        return self

    @classmethod
    def from_label_free_mapping(cls, value: Mapping[str, Any]) -> "GeometryHint":
        reject_supervised_fields(value, path="geometry")
        allowed = {"pivot_xy", "start_xy", "end_xy", "confidence", "source"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown geometry fields: {sorted(unknown)}")
        result = cls(
            pivot_xy=tuple(float(item) for item in value["pivot_xy"]),
            start_xy=tuple(float(item) for item in value["start_xy"]),
            end_xy=tuple(float(item) for item in value["end_xy"]),
            confidence=float(value.get("confidence", 0.0)),
            source=str(value.get("source") or ""),
        )
        return result.validate()


@dataclass(frozen=True)
class OCRToken:
    text: str
    score: float
    box: tuple[tuple[float, float], ...]

    def validate(self) -> "OCRToken":
        if not math.isfinite(float(self.score)) or not 0.0 <= self.score <= 1.0:
            raise ValueError("OCR score must be in [0,1]")
        points = np.asarray(self.box, dtype=np.float64)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("OCR box must contain four finite xy points")
        return self


@dataclass(frozen=True)
class PositionedNumericToken:
    source_index: int
    text: str
    value: float
    score: float
    center_xy: tuple[float, float]
    radius_ratio: float
    progress: float


@dataclass(frozen=True)
class NumericRangePrediction:
    protocol: str
    status: bool
    prediction_space: str
    pred_start: float | None
    pred_end: float | None
    confidence: float
    failure_reason: str | None
    telemetry: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class OCRBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def infer(self, image_bgr: np.ndarray) -> tuple[list[OCRToken], float]: ...


@dataclass(frozen=True)
class GeometryProviderResult:
    hint: GeometryHint
    telemetry: Mapping[str, Any]


@runtime_checkable
class AutomaticGeometryProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def predict(self, image_bgr: np.ndarray) -> GeometryProviderResult: ...


def _angle(point_xy: Sequence[float], pivot_xy: Sequence[float]) -> float:
    return math.atan2(
        float(point_xy[1]) - float(pivot_xy[1]),
        float(point_xy[0]) - float(pivot_xy[0]),
    ) % (2.0 * math.pi)


def directed_arc_radians(geometry: GeometryHint) -> float:
    start = _angle(geometry.start_xy, geometry.pivot_xy)
    end = _angle(geometry.end_xy, geometry.pivot_xy)
    return (end - start) % (2.0 * math.pi)


def directed_progress(
    point_xy: Sequence[float],
    geometry: GeometryHint,
    *,
    endpoint_margin_fraction: float = 0.10,
) -> float | None:
    """Map an xy point to progress on the ordered start-to-end arc.

    OCR boxes can sit a few pixels outside a predicted endpoint.  The short
    complementary gap is therefore represented as a small negative progress
    near start or a progress just above one near end, rather than wrapping a
    start token to approximately 1.1.
    """

    geometry.validate()
    start_angle = _angle(geometry.start_xy, geometry.pivot_xy)
    end_angle = _angle(geometry.end_xy, geometry.pivot_xy)
    point_angle = _angle(point_xy, geometry.pivot_xy)
    arc = (end_angle - start_angle) % (2.0 * math.pi)
    delta = (point_angle - start_angle) % (2.0 * math.pi)
    if delta <= arc:
        return delta / arc

    signed_from_start = math.atan2(
        math.sin(point_angle - start_angle), math.cos(point_angle - start_angle)
    )
    signed_from_end = math.atan2(
        math.sin(point_angle - end_angle), math.cos(point_angle - end_angle)
    )
    margin = arc * float(endpoint_margin_fraction)
    if -margin <= signed_from_start < 0.0:
        return signed_from_start / arc
    if 0.0 < signed_from_end <= margin:
        return 1.0 + signed_from_end / arc
    return None


def parse_numeric_text(text: str) -> float | None:
    """Parse signed decimal scale text while conservatively rejecting units.

    A single comma is normalized only when it separates one or two trailing
    digits (the common OCR confusion for a decimal point).  Longer comma
    groups are ambiguous with thousands separators and are rejected.
    """

    normalized = _normalize_numeric_text(text)
    if normalized is None:
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _normalize_numeric_text(text: str) -> str | None:
    normalized = "".join(str(text).translate(_MINUS_TRANSLATION).split())
    if "," in normalized:
        if "." in normalized or normalized.count(",") != 1:
            return None
        left, right = normalized.split(",")
        signed_left = left[1:] if left[:1] in "+-" else left
        if not signed_left.isdigit() or not right.isdigit() or not 1 <= len(right) <= 2:
            return None
        normalized = f"{left}.{right}"
    if not _NUMERIC_RE.fullmatch(normalized):
        return None
    return normalized


def parse_integer_text(text: str) -> int | None:
    """Backward-compatible strict integer helper used only by audit tooling."""

    value = parse_numeric_text(text)
    if value is None or abs(value - round(value)) > 1e-12:
        return None
    return int(round(value))


def position_numeric_tokens(
    tokens: Sequence[OCRToken],
    *,
    image_shape: Sequence[int],
    geometry: GeometryHint,
    minimum_score: float = 0.20,
    annulus_inner_ratio: float = 0.52,
    annulus_outer_ratio: float = 1.18,
    endpoint_margin_fraction: float = 0.10,
) -> tuple[list[PositionedNumericToken], dict[str, int]]:
    """Apply numeric, annular and predicted-arc filtering without GT boxes."""

    geometry.validate()
    height, width = int(image_shape[0]), int(image_shape[1])
    if min(height, width) < 32:
        raise ValueError("image_shape is invalid")
    pivot = np.asarray(
        [geometry.pivot_xy[0] * width, geometry.pivot_xy[1] * height],
        dtype=np.float64,
    )
    start = np.asarray(
        [geometry.start_xy[0] * width, geometry.start_xy[1] * height],
        dtype=np.float64,
    )
    end = np.asarray(
        [geometry.end_xy[0] * width, geometry.end_xy[1] * height],
        dtype=np.float64,
    )
    reference_radius = 0.5 * (
        float(np.linalg.norm(start - pivot)) + float(np.linalg.norm(end - pivot))
    )
    if reference_radius <= 2.0:
        raise ValueError("geometry reference radius is too small")

    counters = {
        "ocr_total": len(tokens),
        "low_score": 0,
        "non_numeric": 0,
        "box_shape": 0,
        "outside_annulus": 0,
        "outside_arc": 0,
        "accepted_numeric": 0,
    }
    result: list[PositionedNumericToken] = []
    for index, raw in enumerate(tokens):
        token = raw.validate()
        if token.score < minimum_score:
            counters["low_score"] += 1
            continue
        value = parse_numeric_text(token.text)
        if value is None:
            counters["non_numeric"] += 1
            continue
        box = np.asarray(token.box, dtype=np.float64)
        box_width = float(np.ptp(box[:, 0]))
        box_height = float(np.ptp(box[:, 1]))
        if box_width > 0.28 * width or box_height > 0.18 * height:
            counters["box_shape"] += 1
            continue
        center = box.mean(axis=0)
        ratio = float(np.linalg.norm(center - pivot) / reference_radius)
        if not annulus_inner_ratio <= ratio <= annulus_outer_ratio:
            counters["outside_annulus"] += 1
            continue
        point_normalized = (float(center[0] / width), float(center[1] / height))
        progress = directed_progress(
            point_normalized,
            geometry,
            endpoint_margin_fraction=endpoint_margin_fraction,
        )
        if progress is None or not -endpoint_margin_fraction <= progress <= 1.0 + endpoint_margin_fraction:
            counters["outside_arc"] += 1
            continue
        result.append(
            PositionedNumericToken(
                source_index=index,
                text=str(token.text),
                value=float(value),
                score=float(token.score),
                center_xy=(float(center[0]), float(center[1])),
                radius_ratio=ratio,
                progress=float(np.clip(progress, 0.0, 1.0)),
            )
        )
    result.sort(key=lambda item: (item.progress, item.source_index))
    counters["accepted_numeric"] = len(result)
    return result, counters


def _weighted_line_fit(rows: Sequence[PositionedNumericToken]) -> tuple[float, float]:
    progress = np.asarray([row.progress for row in rows], dtype=np.float64)
    values = np.asarray([row.value for row in rows], dtype=np.float64)
    weights = np.asarray([max(row.score, 1e-3) for row in rows], dtype=np.float64)
    design = np.stack((np.ones_like(progress), progress), axis=1)
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * values)
    try:
        intercept, slope = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        intercept, slope = np.linalg.lstsq(
            design * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None
        )[0]
    return float(intercept), float(slope)


def _numeric_quantum(rows: Sequence[PositionedNumericToken]) -> float:
    places = 0
    for row in rows:
        normalized = _normalize_numeric_text(row.text)
        if normalized is not None and "." in normalized:
            places = max(places, len(normalized.rsplit(".", 1)[1]))
    return 10.0 ** (-min(places, 4))


def _residual_limit(candidate_range: float, *, quantum: float) -> float:
    lower = max(1e-5, 1.25 * float(quantum))
    upper = max(lower, 24.0 * float(quantum))
    return float(np.clip(0.0125 * abs(candidate_range), lower, upper))


def _robust_integer_step(values: Sequence[float]) -> int | None:
    if any(abs(float(value) - round(float(value))) > 1e-8 for value in values):
        return None
    unique = sorted(set(int(round(float(value))) for value in values))
    if len(unique) < 3:
        return None
    divisor = 0
    for left, right in zip(unique, unique[1:]):
        divisor = math.gcd(divisor, int(right - left))
    return divisor if divisor > 0 else None


@dataclass(frozen=True)
class ArithmeticGrid:
    step: float
    residue: float
    decimal_places: int


def _robust_arithmetic_grid(
    rows: Sequence[PositionedNumericToken],
) -> ArithmeticGrid | None:
    """Infer a signed decimal AP grid from inlier OCR strings.

    The grid is learned from observed strings and supports values such as
    ``-0.4, -0.2, 0, 0.2``.  It is not an integer-range assumption.
    """

    normalized = [_normalize_numeric_text(row.text) for row in rows]
    if any(value is None for value in normalized):
        return None
    decimal_places = max(
        (len(value.rsplit(".", 1)[1]) if "." in value else 0)
        for value in normalized
        if value is not None
    )
    decimal_places = min(int(decimal_places), 4)
    scale = 10**decimal_places
    scaled: list[int] = []
    for row in rows:
        value = float(row.value) * scale
        if abs(value - round(value)) > 1e-6:
            return None
        scaled.append(int(round(value)))
    unique = sorted(set(scaled))
    if len(unique) < 3:
        return None
    divisor = 0
    for left, right in zip(unique, unique[1:]):
        divisor = math.gcd(divisor, right - left)
    if divisor <= 0:
        return None
    votes: dict[int, float] = {}
    for row, value in zip(rows, scaled, strict=True):
        residue = value % divisor
        votes[residue] = votes.get(residue, 0.0) + row.score
    residue_scaled = max(sorted(votes), key=lambda key: votes[key])
    return ArithmeticGrid(
        step=float(divisor / scale),
        residue=float(residue_scaled / scale),
        decimal_places=decimal_places,
    )


def _snap_to_progression(raw: float, *, residue: int, step: int) -> int:
    rounded = int(round(raw))
    if step <= 0:
        return rounded
    candidate = int(residue + round((raw - residue) / step) * step)
    # A sparse OCR inventory may expose every second label and hide a final
    # half-step.  Snap only when the predicted endpoint is genuinely close.
    if abs(candidate - raw) <= max(1.0, 0.35 * step):
        return candidate
    return rounded


def _snap_to_decimal_grid(raw: float, grid: ArithmeticGrid) -> float:
    candidate = grid.residue + round((raw - grid.residue) / grid.step) * grid.step
    quantum = 10.0 ** (-grid.decimal_places)
    if abs(candidate - raw) <= max(0.5 * quantum, 0.35 * grid.step):
        return float(round(candidate, grid.decimal_places))
    return float(raw)


def decode_numeric_range(
    tokens: Sequence[OCRToken],
    *,
    image_shape: Sequence[int],
    geometry: GeometryHint,
    minimum_inliers: int = 3,
    minimum_progress_span: float = 0.15,
    minimum_pair_span: float = 0.08,
    maximum_abs_range: float = 100_000.0,
) -> NumericRangePrediction:
    """Decode start/end with deterministic two-point weighted RANSAC."""

    positioned, counters = position_numeric_tokens(
        tokens, image_shape=image_shape, geometry=geometry
    )
    base_telemetry: dict[str, Any] = {
        "geometry": asdict(geometry),
        "filter_counts": counters,
        "numeric_tokens": [asdict(row) for row in positioned],
        "decoder": "deterministic_all_pairs_weighted_ransac",
        "minimum_inliers": int(minimum_inliers),
    }
    if len(positioned) < minimum_inliers:
        return NumericRangePrediction(
            PROTOCOL, False, PREDICTION_SPACE, None, None, 0.0,
            "fewer_than_three_annular_numeric_tokens", base_telemetry,
        )

    progress = np.asarray([row.progress for row in positioned], dtype=np.float64)
    values = np.asarray([row.value for row in positioned], dtype=np.float64)
    weights = np.asarray([row.score for row in positioned], dtype=np.float64)
    quantum = _numeric_quantum(positioned)
    best: tuple[tuple[float, ...], np.ndarray, float, float, float] | None = None
    for left in range(len(positioned)):
        for right in range(left + 1, len(positioned)):
            delta_progress = float(progress[right] - progress[left])
            if abs(delta_progress) < minimum_pair_span:
                continue
            slope = float((values[right] - values[left]) / delta_progress)
            if not 1e-6 <= slope <= maximum_abs_range:
                continue
            intercept = float(values[left] - slope * progress[left])
            limit = _residual_limit(slope, quantum=quantum)
            residual = np.abs(values - (intercept + slope * progress))
            inliers = residual <= limit
            if int(inliers.sum()) < minimum_inliers:
                continue
            inlier_progress = progress[inliers]
            span = float(inlier_progress.max() - inlier_progress.min())
            if span < minimum_progress_span:
                continue
            ordered_values = values[inliers][np.argsort(inlier_progress)]
            if bool(np.any(np.diff(ordered_values) < 0.0)):
                continue
            weighted_mass = float(weights[inliers].sum())
            rmse = float(
                np.sqrt(np.average(np.square(residual[inliers]), weights=weights[inliers]))
            )
            score = weighted_mass + 2.0 * span - 0.75 * (rmse / limit)
            tie_break = (
                score,
                float(inliers.sum()),
                span,
                -rmse,
                -float(left),
                -float(right),
            )
            if best is None or tie_break > best[0]:
                best = (tie_break, inliers, intercept, slope, limit)

    if best is None:
        return NumericRangePrediction(
            PROTOCOL, False, PREDICTION_SPACE, None, None, 0.0,
            "no_positive_monotonic_ransac_consensus", base_telemetry,
        )

    _, inliers, intercept, slope, limit = best
    # Two deterministic refinement passes remove pair-specific bias.
    for _ in range(2):
        selected = [row for row, keep in zip(positioned, inliers, strict=True) if keep]
        intercept, slope = _weighted_line_fit(selected)
        limit = _residual_limit(slope, quantum=quantum)
        residual = np.abs(values - (intercept + slope * progress))
        candidate = residual <= limit
        if int(candidate.sum()) < minimum_inliers:
            break
        ordered = np.argsort(progress[candidate])
        if bool(np.any(np.diff(values[candidate][ordered]) < 0.0)):
            break
        inliers = candidate

    selected = [row for row, keep in zip(positioned, inliers, strict=True) if keep]
    intercept, slope = _weighted_line_fit(selected)
    residual = np.abs(values - (intercept + slope * progress))
    selected_residual = residual[inliers]
    selected_weights = weights[inliers]
    rmse = float(np.sqrt(np.average(np.square(selected_residual), weights=selected_weights)))
    span = float(progress[inliers].max() - progress[inliers].min())
    step = _robust_integer_step([row.value for row in selected])
    arithmetic_grid = _robust_arithmetic_grid(selected)
    raw_start, raw_end = float(intercept), float(intercept + slope)
    integer_diagnostic: dict[str, Any] | None = None
    if step is not None:
        residue_votes: dict[int, float] = {}
        for row in selected:
            residue = int(round(row.value)) % step
            residue_votes[residue] = residue_votes.get(residue, 0.0) + row.score
        residue = max(sorted(residue_votes), key=lambda key: residue_votes[key])
        integer_diagnostic = {
            "scope": "optional SyncG-integer evaluation diagnostic; not deployment output",
            "pred_start": _snap_to_progression(raw_start, residue=residue, step=step),
            "pred_end": _snap_to_progression(raw_end, residue=residue, step=step),
            "step": step,
            "residue": residue,
        }
    else:
        residue = None

    if arithmetic_grid is not None:
        pred_start = _snap_to_decimal_grid(raw_start, arithmetic_grid)
        pred_end = _snap_to_decimal_grid(raw_end, arithmetic_grid)
    else:
        pred_start, pred_end = raw_start, raw_end

    if pred_end <= pred_start:
        return NumericRangePrediction(
            PROTOCOL, False, PREDICTION_SPACE, None, None, 0.0,
            "decoded_range_is_not_positive", base_telemetry,
        )

    inlier_mass_fraction = float(selected_weights.sum() / max(weights.sum(), 1e-9))
    span_score = float(np.clip(span, 0.0, 1.0))
    residual_score = math.exp(
        -rmse / max(_residual_limit(slope, quantum=quantum), 1e-9)
    )
    count_score = min(1.0, len(selected) / 6.0)
    decoder_confidence = (
        0.30 * inlier_mass_fraction
        + 0.25 * span_score
        + 0.25 * residual_score
        + 0.20 * count_score
    )
    confidence = float(
        np.clip(0.85 * decoder_confidence + 0.15 * geometry.confidence, 0.0, 1.0)
    )
    base_telemetry.update(
        {
            "fit": {
                "raw_start": raw_start,
                "raw_end": raw_end,
                "raw_range": slope,
                "predicted_integer_step": step,
                "predicted_integer_residue": residue,
                "syncg_integer_progression_diagnostic": integer_diagnostic,
                "arithmetic_progression_grid": (
                    asdict(arithmetic_grid) if arithmetic_grid is not None else None
                ),
                "deployment_grid_supports_signed_decimals": True,
                "residual_limit": _residual_limit(slope, quantum=quantum),
                "observed_numeric_quantum": quantum,
                "weighted_rmse": rmse,
                "progress_span": span,
                "inlier_count": len(selected),
                "inlier_source_indices": [row.source_index for row in selected],
                "inlier_values": [row.value for row in selected],
                "inlier_mass_fraction": inlier_mass_fraction,
            },
            "decoder_confidence": decoder_confidence,
        }
    )
    return NumericRangePrediction(
        PROTOCOL,
        True,
        PREDICTION_SPACE,
        pred_start,
        pred_end,
        confidence,
        None,
        base_telemetry,
    )


class FrozenRapidOCRBackend:
    """CPU-only frozen RapidOCR adapter; it never downloads model files."""

    MODEL_NAMES: Final[tuple[str, ...]] = (
        "ch_PP-OCRv4_det_infer.onnx",
        "ch_PP-OCRv4_rec_infer.onnx",
        "ch_ppocr_mobile_v2.0_cls_infer.onnx",
        "ppocr_keys_v1.txt",
    )

    def __init__(self, runtime_root: Path = DEFAULT_OCR_RUNTIME, *, cpu_threads: int = 4):
        root = Path(runtime_root).resolve(strict=True)
        if not (root / "rapidocr" / "__init__.py").is_file():
            raise FileNotFoundError(f"RapidOCR package is absent under {root}")
        models = root / "rapidocr" / "models"
        bindings = {
            name: {
                "path": str((models / name).resolve(strict=True)),
                "sha256": sha256_file(models / name),
            }
            for name in self.MODEL_NAMES
        }
        if str(root) not in sys.path:
            # Append so the pinned research environment keeps its NumPy/OpenCV;
            # only missing RapidOCR/ONNX dependencies resolve from this target.
            sys.path.append(str(root))
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        params = {
            "Global.log_level": "error",
            "Global.text_score": 0.20,
            "Global.max_side_len": 2000,
            "Det.box_thresh": 0.20,
            "Det.unclip_ratio": 1.30,
            "EngineConfig.onnxruntime.intra_op_num_threads": int(cpu_threads),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
        }
        started = time.perf_counter()
        self._engine = RapidOCR(params=params)
        self._initialization_seconds = time.perf_counter() - started
        self._identity = {
            "backend": "RapidOCR",
            "version": "3.8.1",
            "execution_provider": "onnxruntime_cpu",
            "runtime_root": str(root),
            "model_bindings": bindings,
            "parameters": params,
            "initialization_seconds": self._initialization_seconds,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def infer(self, image_bgr: np.ndarray) -> tuple[list[OCRToken], float]:
        image = validate_canonical_roi(image_bgr)
        started = time.perf_counter()
        output = self._engine(image)
        elapsed = time.perf_counter() - started
        if output.boxes is None or output.txts is None or output.scores is None:
            return [], elapsed
        tokens = [
            OCRToken(
                text=str(text),
                score=float(score),
                box=tuple((float(point[0]), float(point[1])) for point in box),
            ).validate()
            for box, text, score in zip(
                output.boxes, output.txts, output.scores, strict=True
            )
        ]
        return tokens, elapsed


class AutomaticNumericRangeEstimator:
    """Component-only OCR/decoder that requires internally predicted geometry.

    Do not expose this class as the paper/deployment entry point.  The primary
    adapter is :class:`AutomaticNumericRangePipeline`, whose ``predict`` method
    accepts only the whole canonical ROI and obtains geometry internally.
    """

    def __init__(self, backend: OCRBackend, *, input_size: int = 768):
        if not isinstance(backend, OCRBackend):
            raise TypeError("backend does not implement OCRBackend")
        if int(input_size) not in (512, 768):
            raise ValueError("numeric range OCR input_size must be 512 or 768")
        self.backend = backend
        self.input_size = int(input_size)

    def predict(
        self, canonical_meter_roi_bgr: np.ndarray, geometry: GeometryHint
    ) -> NumericRangePrediction:
        image = validate_canonical_roi(canonical_meter_roi_bgr)
        geometry.validate()
        interpolation = (
            cv2.INTER_AREA
            if max(image.shape[:2]) > self.input_size
            else cv2.INTER_CUBIC
        )
        high_resolution = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=interpolation,
        )
        tokens, ocr_seconds = self.backend.infer(high_resolution)
        prediction = decode_numeric_range(
            tokens,
            image_shape=high_resolution.shape,
            geometry=geometry,
        )
        telemetry = dict(prediction.telemetry)
        telemetry.update(
            {
                "whole_roi_direct_resize": True,
                "second_crop_or_detector": False,
                "ocr_input_shape": list(high_resolution.shape),
                "ocr_seconds": float(ocr_seconds),
                "ocr_backend": dict(self.backend.identity),
                "canonical_roi_sha256": image_sha256(image),
            }
        )
        return NumericRangePrediction(
            protocol=prediction.protocol,
            status=prediction.status,
            prediction_space=prediction.prediction_space,
            pred_start=prediction.pred_start,
            pred_end=prediction.pred_end,
            confidence=prediction.confidence,
            failure_reason=prediction.failure_reason,
            telemetry=telemetry,
        )


class AutomaticNumericRangePipeline:
    """Primary label-free adapter: one whole ROI in, numeric start/end out."""

    def __init__(
        self,
        geometry_provider: AutomaticGeometryProvider,
        ocr_backend: OCRBackend,
        *,
        input_size: int = 768,
    ):
        if not isinstance(geometry_provider, AutomaticGeometryProvider):
            raise TypeError("geometry_provider does not implement AutomaticGeometryProvider")
        self.geometry_provider = geometry_provider
        self.component = AutomaticNumericRangeEstimator(ocr_backend, input_size=input_size)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "protocol": PROTOCOL,
            "primary_input": "one whole canonical meter ROI (BGR uint8)",
            "manual_geometry_input": False,
            "geometry_provider": dict(self.geometry_provider.identity),
            "ocr_backend": dict(self.component.backend.identity),
            "ocr_input_size": self.component.input_size,
        }

    def predict(self, canonical_meter_roi_bgr: np.ndarray) -> NumericRangePrediction:
        image = validate_canonical_roi(canonical_meter_roi_bgr)
        input_digest = image_sha256(image)
        provided = self.geometry_provider.predict(image)
        if not isinstance(provided, GeometryProviderResult):
            raise TypeError("automatic geometry provider returned an invalid result")
        prediction = self.component.predict(image, provided.hint)
        telemetry = dict(prediction.telemetry)
        telemetry.update(
            {
                "primary_adapter": {
                    "accepts_manual_geometry": False,
                    "accepts_reference_packet": False,
                    "accepts_physical_scale_values": False,
                    "geometry_provider_binding": dict(self.geometry_provider.identity),
                    "geometry_telemetry": dict(provided.telemetry),
                    "input_image_sha256": input_digest,
                    "geometry_and_ocr_same_image_sha256": input_digest,
                }
            }
        )
        return NumericRangePrediction(
            protocol=prediction.protocol,
            status=prediction.status,
            prediction_space=prediction.prediction_space,
            pred_start=prediction.pred_start,
            pred_end=prediction.pred_end,
            confidence=prediction.confidence,
            failure_reason=prediction.failure_reason,
            telemetry=telemetry,
        )


def direct_endpoint_baseline(
    prediction: NumericRangePrediction,
    *,
    maximum_endpoint_distance: float = 0.12,
) -> tuple[float | None, float | None]:
    """Diagnostic only: nearest raw numeric token to each predicted endpoint."""

    raw = prediction.telemetry.get("numeric_tokens") or []
    rows = [row for row in raw if isinstance(row, Mapping)]
    if not rows:
        return None, None
    start_row = min(rows, key=lambda row: abs(float(row["progress"])))
    end_row = min(rows, key=lambda row: abs(float(row["progress"]) - 1.0))
    start = (
        float(start_row["value"])
        if abs(float(start_row["progress"])) <= maximum_endpoint_distance
        else None
    )
    end = (
        float(end_row["value"])
        if abs(float(end_row["progress"]) - 1.0) <= maximum_endpoint_distance
        else None
    )
    return start, end
