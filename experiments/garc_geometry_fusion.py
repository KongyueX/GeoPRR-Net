"""Image-only covariance fusion for the geometry consumed by GARC.

This module is deliberately a *provider*, not another learned model.  It
combines already frozen visual components on the same canonical meter ROI:

* V5 ScaleMark geometry (pivot, ordered start and end points);
* an optional Base Mask--Geometry hypothesis; and
* PEPD pointer direction, pivot and native angular uncertainty.

The public boundary remains one BGR ROI.  Numeric range values, manual points,
reference packets and labels cannot be supplied.  Fusion is performed in the
normalized direct-square model frame so that a PEPD pivot predicted on a
256x256 resize is never divided by the dimensions of a non-square source ROI.

The algorithm differs from simple V5 -> OCR serial composition in three ways:

1. pivot estimates are combined by capped precision (inverse-covariance)
   weighting, which avoids counting the correlated PEPD/V5 backbone twice;
2. endpoint rays are averaged on the circle, so 359 and 1 degrees remain
   neighbours; and
3. a confident PEPD ray resolves a single arc-branch ambiguity, while
   unresolved strong conflicts cause an explicit abstention.

No dataset access, training or GPU work occurs in this module.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from experiments.automatic_numeric_range import (
    AutomaticGeometryProvider,
    GeometryHint,
    GeometryProviderResult,
    image_sha256,
    reject_supervised_fields,
    validate_canonical_roi,
)


PROTOCOL: Final[str] = "garc_covariance_geometry_fusion_v1"
COORDINATE_FRAME: Final[str] = (
    "normalized direct-square model frame; x/y are normalized before angular fusion"
)
_TAU: Final[float] = 2.0 * math.pi


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _angle(point: Sequence[float], pivot: Sequence[float]) -> float:
    return math.atan2(
        float(point[1]) - float(pivot[1]),
        float(point[0]) - float(pivot[0]),
    ) % _TAU


def _circular_distance(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def _directed_arc(start: float, end: float) -> float:
    return (end - start) % _TAU


def _circular_mean(angles: Sequence[float], weights: Sequence[float]) -> float:
    if len(angles) != len(weights) or not angles:
        raise ValueError("circular mean requires aligned non-empty values")
    sine = sum(float(weight) * math.sin(float(angle)) for angle, weight in zip(angles, weights, strict=True))
    cosine = sum(float(weight) * math.cos(float(angle)) for angle, weight in zip(angles, weights, strict=True))
    if math.hypot(sine, cosine) <= 1e-12:
        raise GeometryFusionRejected("circular_endpoint_mean_collapsed")
    return math.atan2(sine, cosine) % _TAU


def _mapping_path(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _point(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    result = np.asarray(value[:2], dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        return None
    return result


def _covariance(value: Any, *, fallback_sigma: float) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64) if value is not None else np.empty((0, 0))
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        return np.eye(2, dtype=np.float64) * float(fallback_sigma) ** 2
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(1e-8, (0.2 * float(fallback_sigma)) ** 2)
    ceiling = max(floor, (5.0 * float(fallback_sigma)) ** 2)
    eigenvalues = np.clip(eigenvalues, floor, ceiling)
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T


def _angle_std_from_telemetry(
    telemetry: Mapping[str, Any],
    *,
    index: int,
    confidence: float,
    config: "GeometryFusionConfig",
) -> float:
    explicit_keys = (
        "start_angle_std_degrees" if index == 0 else "end_angle_std_degrees",
        "endpoint_angle_std_degrees",
    )
    explicit = _finite(telemetry.get(explicit_keys[0]))
    if explicit is None:
        paired = telemetry.get(explicit_keys[1])
        if isinstance(paired, (list, tuple, np.ndarray)) and len(paired) > index:
            explicit = _finite(paired[index])
    if explicit is not None and explicit > 0.0:
        return float(
            np.clip(
                explicit,
                config.minimum_endpoint_std_degrees,
                config.maximum_endpoint_std_degrees,
            )
        )

    entropy = telemetry.get("endpoint_entropy")
    entropy_value: float | None = None
    if isinstance(entropy, (list, tuple, np.ndarray)) and len(entropy) > index:
        entropy_value = _finite(entropy[index])
    uncertainty = _finite(telemetry.get("uncertainty_score"))
    proxy = (
        _clip01(entropy_value)
        if entropy_value is not None
        else (_clip01(uncertainty) if uncertainty is not None else 1.0 - confidence)
    )
    return float(
        config.minimum_endpoint_std_degrees
        + proxy
        * (config.maximum_endpoint_std_degrees - config.minimum_endpoint_std_degrees)
    )


@dataclass(frozen=True)
class GeometryFusionConfig:
    """Frozen label-free thresholds for the low-cost fusion screen."""

    minimum_geometry_confidence: float = 0.10
    strong_geometry_confidence: float = 0.55
    minimum_endpoint_std_degrees: float = 1.5
    maximum_endpoint_std_degrees: float = 24.0
    minimum_pivot_std_fraction: float = 0.006
    maximum_pivot_std_fraction: float = 0.060
    maximum_endpoint_disagreement_degrees: float = 24.0
    maximum_arc_disagreement_degrees: float = 34.0
    maximum_pivot_distance_fraction: float = 0.12
    maximum_pivot_mahalanobis: float = 5.0
    strong_pepd_std_degrees: float = 16.0
    pointer_branch_inside_probability: float = 0.75
    pointer_branch_outside_probability: float = 0.25
    maximum_confident_pointer_outside_degrees: float = 14.0
    pepd_pivot_precision_cap_fraction: float = 0.50
    single_geometry_confidence_penalty: float = 0.84
    missing_pepd_confidence_penalty: float = 0.88

    def validate(self) -> "GeometryFusionConfig":
        probabilities = (
            self.minimum_geometry_confidence,
            self.strong_geometry_confidence,
            self.pointer_branch_inside_probability,
            self.pointer_branch_outside_probability,
            self.pepd_pivot_precision_cap_fraction,
            self.single_geometry_confidence_penalty,
            self.missing_pepd_confidence_penalty,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("fusion probability/penalty values must be in [0,1]")
        if self.minimum_geometry_confidence >= self.strong_geometry_confidence:
            raise ValueError("strong geometry threshold must exceed minimum threshold")
        if self.pointer_branch_outside_probability >= self.pointer_branch_inside_probability:
            raise ValueError("branch probabilities are not ordered")
        positive = (
            self.minimum_endpoint_std_degrees,
            self.maximum_endpoint_std_degrees,
            self.minimum_pivot_std_fraction,
            self.maximum_pivot_std_fraction,
            self.maximum_endpoint_disagreement_degrees,
            self.maximum_arc_disagreement_degrees,
            self.maximum_pivot_distance_fraction,
            self.maximum_pivot_mahalanobis,
            self.strong_pepd_std_degrees,
            self.maximum_confident_pointer_outside_degrees,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("fusion scale/threshold values must be finite and positive")
        if self.minimum_endpoint_std_degrees >= self.maximum_endpoint_std_degrees:
            raise ValueError("endpoint uncertainty bounds are not ordered")
        if self.minimum_pivot_std_fraction >= self.maximum_pivot_std_fraction:
            raise ValueError("pivot uncertainty bounds are not ordered")
        return self


class GeometryFusionRejected(RuntimeError):
    """Controlled abstention when image-derived geometry cannot be reconciled."""

    def __init__(
        self,
        code: str,
        *,
        telemetry: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)[:128]
        self.telemetry = dict(telemetry or {})
        super().__init__(self.code)


@runtime_checkable
class ImageOnlyPEPDStructuralProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _GeometryObservation:
    name: str
    hint: GeometryHint
    pivot_covariance: np.ndarray
    start_angle: float
    end_angle: float
    arc: float
    start_radius: float
    end_radius: float
    start_std_radians: float
    end_std_radians: float
    telemetry: Mapping[str, Any]

    @property
    def confidence(self) -> float:
        return float(self.hint.confidence)


@dataclass(frozen=True)
class _PEPDObservation:
    available: bool
    pivot_xy: np.ndarray | None
    pivot_covariance: np.ndarray | None
    pointer_angle: float | None
    angle_std_radians: float
    confidence: float
    telemetry: Mapping[str, Any]
    failure_code: str | None = None


def _geometry_observation(
    name: str,
    result: GeometryProviderResult,
    config: GeometryFusionConfig,
) -> _GeometryObservation:
    if not isinstance(result, GeometryProviderResult):
        raise TypeError("geometry provider returned an invalid result")
    hint = result.hint.validate()
    if hint.confidence < config.minimum_geometry_confidence:
        raise GeometryFusionRejected(f"{name}_confidence_below_floor")
    telemetry = dict(result.telemetry)
    reject_supervised_fields(telemetry, path=f"{name}_telemetry")
    confidence = float(hint.confidence)
    fallback_sigma = (
        config.minimum_pivot_std_fraction
        + (1.0 - confidence)
        * (config.maximum_pivot_std_fraction - config.minimum_pivot_std_fraction)
    )
    covariance_value = telemetry.get("pivot_covariance_normalized")
    if covariance_value is None:
        covariance_value = _mapping_path(telemetry, "uncertainty", "pivot_covariance_normalized")
    pivot_covariance = _covariance(covariance_value, fallback_sigma=fallback_sigma)
    pivot = np.asarray(hint.pivot_xy, dtype=np.float64)
    start = np.asarray(hint.start_xy, dtype=np.float64)
    end = np.asarray(hint.end_xy, dtype=np.float64)
    start_angle = _angle(start, pivot)
    end_angle = _angle(end, pivot)
    return _GeometryObservation(
        name=name,
        hint=hint,
        pivot_covariance=pivot_covariance,
        start_angle=start_angle,
        end_angle=end_angle,
        arc=_directed_arc(start_angle, end_angle),
        start_radius=float(np.linalg.norm(start - pivot)),
        end_radius=float(np.linalg.norm(end - pivot)),
        start_std_radians=math.radians(
            _angle_std_from_telemetry(
                telemetry, index=0, confidence=confidence, config=config
            )
        ),
        end_std_radians=math.radians(
            _angle_std_from_telemetry(
                telemetry, index=1, confidence=confidence, config=config
            )
        ),
        telemetry=telemetry,
    )


def _pepd_telemetry(record: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = record.get("telemetry")
    if isinstance(direct, Mapping):
        return direct
    nested = _mapping_path(record, "expert_results", "pepd", "telemetry")
    return nested if isinstance(nested, Mapping) else {}


def _pepd_observation(
    record: Mapping[str, Any],
    config: GeometryFusionConfig,
) -> _PEPDObservation:
    reject_supervised_fields(record, path="pepd_structural_output")
    status_value = record.get("status")
    status = status_value is True or str(status_value).casefold() in {
        "ok",
        "success",
        "complete",
    }
    telemetry = dict(_pepd_telemetry(record))
    direction = _point(telemetry.get("direction_xy"))
    pointer_angle: float | None = None
    if direction is not None and float(np.linalg.norm(direction)) > 1e-12:
        direction /= float(np.linalg.norm(direction))
        pointer_angle = math.atan2(float(direction[1]), float(direction[0])) % _TAU
    else:
        production_angle = _finite(telemetry.get("pointer_angle"))
        if production_angle is not None:
            # image_angle_from_direction(theta) = mathematical_image_theta - 90 deg.
            pointer_angle = math.radians((production_angle + 90.0) % 360.0)

    pivot = _point(telemetry.get("pivot_xy_normalized"))
    if pivot is None:
        pivot_input = _point(telemetry.get("pivot_input_xy"))
        native_shape = telemetry.get("native_input_shape")
        native_size = _finite(telemetry.get("native_input_size"))
        if pivot_input is not None and isinstance(native_shape, (list, tuple)) and len(native_shape) >= 2:
            native_height = _finite(native_shape[0])
            native_width = _finite(native_shape[1])
            if native_height and native_width and native_height > 1.0 and native_width > 1.0:
                pivot = np.asarray(
                    [pivot_input[0] / native_width, pivot_input[1] / native_height],
                    dtype=np.float64,
                )
        elif pivot_input is not None and native_size is not None and native_size > 1.0:
            # Crucial for a non-square caller ROI: these coordinates belong to
            # the PEPD square resize, not to caller width/height.
            pivot = pivot_input / native_size
    if pivot is not None and (
        not np.isfinite(pivot).all() or np.any(pivot < -0.05) or np.any(pivot > 1.05)
    ):
        pivot = None

    angle_std_degrees = _finite(telemetry.get("angle_std_degrees"))
    entropy = _finite(telemetry.get("angle_bin_entropy"))
    resultant = _finite(telemetry.get("angle_bin_resultant_length"))
    if angle_std_degrees is None or angle_std_degrees <= 0.0:
        entropy_proxy = 1.0 if entropy is None else _clip01(entropy)
        angle_std_degrees = 5.0 + 40.0 * entropy_proxy
    angle_std_degrees = float(np.clip(angle_std_degrees, 0.5, 90.0))
    angle_confidence = math.exp(-0.5 * (angle_std_degrees / 24.0) ** 2)
    if entropy is not None:
        angle_confidence *= 1.0 - 0.45 * _clip01(entropy)
    if resultant is not None:
        angle_confidence *= 0.55 + 0.45 * _clip01(resultant)
    angle_confidence = _clip01(angle_confidence)

    pivot_covariance: np.ndarray | None = None
    if pivot is not None:
        pivot_peak = _finite(telemetry.get("pivot_peak"))
        pivot_entropy = _finite(telemetry.get("pivot_spatial_entropy"))
        uncertainty_proxy = 0.5
        if pivot_entropy is not None:
            uncertainty_proxy = _clip01(pivot_entropy)
        if pivot_peak is not None:
            uncertainty_proxy = _clip01(0.65 * uncertainty_proxy + 0.35 * (1.0 - _clip01(pivot_peak)))
        fallback_sigma = (
            config.minimum_pivot_std_fraction
            + uncertainty_proxy
            * (config.maximum_pivot_std_fraction - config.minimum_pivot_std_fraction)
        )
        pivot_covariance = _covariance(
            telemetry.get("pivot_covariance_normalized"),
            fallback_sigma=fallback_sigma,
        )

    available = bool(status and pointer_angle is not None)
    failure = None if available else str(record.get("failure_code") or "pepd_direction_unavailable")[:128]
    return _PEPDObservation(
        available=available,
        pivot_xy=pivot,
        pivot_covariance=pivot_covariance,
        pointer_angle=pointer_angle if available else None,
        angle_std_radians=math.radians(angle_std_degrees),
        confidence=angle_confidence if available else 0.0,
        telemetry={
            "angle_std_degrees": angle_std_degrees,
            "direction_confidence": angle_confidence if available else 0.0,
            "pivot_available": pivot is not None,
            "pivot_source": (
                "normalized_or_native_square_coordinates" if pivot is not None else "unavailable"
            ),
        },
        failure_code=failure,
    )


def _pointer_consistency(
    observation: _GeometryObservation,
    pepd: _PEPDObservation,
) -> dict[str, Any]:
    if not pepd.available or pepd.pointer_angle is None:
        return {
            "available": False,
            "inside_arc": None,
            "progress": None,
            "outside_degrees": None,
            "probability": None,
        }
    delta = (pepd.pointer_angle - observation.start_angle) % _TAU
    if delta <= observation.arc:
        outside = 0.0
        progress = delta / observation.arc
    else:
        outside = min(
            _circular_distance(pepd.pointer_angle, observation.start_angle),
            _circular_distance(pepd.pointer_angle, observation.end_angle),
        )
        progress = None
    sigma = max(pepd.angle_std_radians, math.radians(1.0))
    probability = math.exp(-0.5 * (outside / sigma) ** 2)
    return {
        "available": True,
        "inside_arc": outside <= 1e-12,
        "progress": None if progress is None else float(progress),
        "outside_degrees": math.degrees(outside),
        "probability": float(probability),
    }


def _pair_diagnostics(
    first: _GeometryObservation,
    second: _GeometryObservation,
    config: GeometryFusionConfig,
) -> dict[str, Any]:
    delta = np.asarray(first.hint.pivot_xy, dtype=np.float64) - np.asarray(
        second.hint.pivot_xy, dtype=np.float64
    )
    covariance = first.pivot_covariance + second.pivot_covariance
    mahalanobis = float(math.sqrt(max(0.0, float(delta @ np.linalg.pinv(covariance) @ delta))))
    start_degrees = math.degrees(_circular_distance(first.start_angle, second.start_angle))
    end_degrees = math.degrees(_circular_distance(first.end_angle, second.end_angle))
    arc_degrees = math.degrees(abs(first.arc - second.arc))
    pivot_distance = float(np.linalg.norm(delta))
    pivot_conflict = bool(
        pivot_distance > config.maximum_pivot_distance_fraction
        and mahalanobis > config.maximum_pivot_mahalanobis
    )
    compatible = bool(
        start_degrees <= config.maximum_endpoint_disagreement_degrees
        and end_degrees <= config.maximum_endpoint_disagreement_degrees
        and arc_degrees <= config.maximum_arc_disagreement_degrees
        and not pivot_conflict
    )
    return {
        "compatible": compatible,
        "start_disagreement_degrees": start_degrees,
        "end_disagreement_degrees": end_degrees,
        "arc_disagreement_degrees": arc_degrees,
        "pivot_distance_fraction": pivot_distance,
        "pivot_mahalanobis": mahalanobis,
        "pivot_conflict": pivot_conflict,
    }


def _geometry_weight(confidence: float, std_radians: float) -> float:
    return max(1e-6, float(confidence)) / max(math.radians(0.5) ** 2, std_radians**2)


def _precision_fuse(
    observations: Sequence[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if not observations:
        raise GeometryFusionRejected("no_pivot_observation")
    precision = np.zeros((2, 2), dtype=np.float64)
    weighted = np.zeros(2, dtype=np.float64)
    for point, covariance, scale in observations:
        contribution = max(1e-8, float(scale)) * np.linalg.pinv(covariance)
        precision += contribution
        weighted += contribution @ point
    covariance = np.linalg.pinv(precision)
    return covariance @ weighted, covariance


def _bounded_endpoint(
    pivot: np.ndarray,
    angle: float,
    radius: float,
) -> tuple[float, float]:
    direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    maximum = float("inf")
    for coordinate, component in zip(pivot, direction, strict=True):
        if component > 1e-12:
            maximum = min(maximum, (1.04 - float(coordinate)) / float(component))
        elif component < -1e-12:
            maximum = min(maximum, (-0.04 - float(coordinate)) / float(component))
    bounded_radius = min(float(radius), 0.98 * maximum)
    if not math.isfinite(bounded_radius) or bounded_radius <= 0.02:
        raise GeometryFusionRejected("fused_endpoint_radius_collapsed")
    endpoint = pivot + bounded_radius * direction
    return float(endpoint[0]), float(endpoint[1])


class GARCCovarianceGeometryFusionProvider:
    """Plug-in ``AutomaticGeometryProvider`` with a one-image public API."""

    def __init__(
        self,
        v5_geometry_provider: AutomaticGeometryProvider,
        pepd_structural_provider: ImageOnlyPEPDStructuralProvider,
        *,
        mask_geometry_provider: AutomaticGeometryProvider | None = None,
        config: GeometryFusionConfig | None = None,
    ) -> None:
        if not isinstance(v5_geometry_provider, AutomaticGeometryProvider):
            raise TypeError("v5_geometry_provider must implement AutomaticGeometryProvider")
        if not isinstance(pepd_structural_provider, ImageOnlyPEPDStructuralProvider):
            raise TypeError("pepd provider must implement the image-only structural interface")
        if mask_geometry_provider is not None and not isinstance(
            mask_geometry_provider, AutomaticGeometryProvider
        ):
            raise TypeError("mask provider must implement AutomaticGeometryProvider")
        self.v5_geometry_provider = v5_geometry_provider
        self.mask_geometry_provider = mask_geometry_provider
        self.pepd_structural_provider = pepd_structural_provider
        self.config = (config or GeometryFusionConfig()).validate()
        self._lock = threading.Lock()
        self._identity = {
            "protocol": PROTOCOL,
            "provider": "garc_covariance_geometry_fusion",
            "primary_input": "one unchanged canonical meter ROI BGR uint8",
            "coordinate_frame": COORDINATE_FRAME,
            "v5_geometry_provider": dict(v5_geometry_provider.identity),
            "mask_geometry_provider": (
                None
                if mask_geometry_provider is None
                else dict(mask_geometry_provider.identity)
            ),
            "pepd_structural_provider": dict(pepd_structural_provider.identity),
            "config": asdict(self.config),
            "fusion": {
                "pivot": "capped inverse-covariance weighting",
                "endpoint_rays": "confidence/uncertainty weighted circular mean",
                "arc_branch": "PEPD uncertainty-aware selection or abstention",
                "single_component_failure": "degraded-confidence fallback",
            },
            "caller_supplied_points_allowed": False,
            "caller_supplied_reference_allowed": False,
            "caller_supplied_numeric_range_allowed": False,
            "external_models_modified": False,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    @classmethod
    def from_existing_base_mask_geometry(
        cls,
        v5_geometry_provider: AutomaticGeometryProvider,
        pepd_structural_provider: ImageOnlyPEPDStructuralProvider,
        *,
        base_mask_geometry_adapter: Any,
        config: GeometryFusionConfig | None = None,
    ) -> "GARCCovarianceGeometryFusionProvider":
        """Wire the repository's real Base Mask--Geometry into fusion.

        The local import avoids coupling the generic fusion arithmetic to the
        legacy runtime loader.  The concrete adapter itself verifies the
        strict both-endpoint production branch and refuses legacy defaults.
        """

        from experiments.garc_base_mask_geometry_provider import (
            ExistingBaseMaskGeometryProvider,
        )

        return cls(
            v5_geometry_provider,
            pepd_structural_provider,
            mask_geometry_provider=ExistingBaseMaskGeometryProvider(
                base_mask_geometry_adapter
            ),
            config=config,
        )

    def _call_geometry(
        self,
        name: str,
        provider: AutomaticGeometryProvider,
        image: np.ndarray,
        input_digest: str,
    ) -> tuple[_GeometryObservation | None, str | None]:
        component_image = image.copy()
        try:
            result = provider.predict(component_image)
            if image_sha256(component_image) != input_digest:
                raise RuntimeError("component_mutated_input")
            return _geometry_observation(name, result, self.config), None
        except GeometryFusionRejected as exc:
            return None, exc.code
        except Exception as exc:  # Component failure is a candidate failure.
            declared = getattr(exc, "code", None)
            if declared:
                return None, f"{name}_{str(declared)[:96]}"
            return None, f"{name}_provider_{type(exc).__name__}"

    def _call_pepd(
        self,
        image: np.ndarray,
        input_digest: str,
    ) -> _PEPDObservation:
        component_image = image.copy()
        try:
            record = self.pepd_structural_provider.predict(
                component_image,
                input_is_canonical_meter_roi=True,
            )
            if image_sha256(component_image) != input_digest:
                raise RuntimeError("component_mutated_input")
            if not isinstance(record, Mapping):
                raise TypeError("PEPD structural output is not a mapping")
            return _pepd_observation(record, self.config)
        except Exception as exc:
            return _PEPDObservation(
                available=False,
                pivot_xy=None,
                pivot_covariance=None,
                pointer_angle=None,
                angle_std_radians=math.radians(90.0),
                confidence=0.0,
                telemetry={},
                failure_code=f"pepd_provider_{type(exc).__name__}",
            )

    def _select_geometry(
        self,
        observations: list[_GeometryObservation],
        pepd: _PEPDObservation,
    ) -> tuple[list[_GeometryObservation], dict[str, Any]]:
        pointer = {
            observation.name: _pointer_consistency(observation, pepd)
            for observation in observations
        }
        if len(observations) == 1:
            return observations, {
                "mode": f"single_{observations[0].name}_fallback",
                "pair": None,
                "pointer_consistency": pointer,
                "discarded": [],
            }
        first, second = observations
        pair = _pair_diagnostics(first, second, self.config)
        if pair["compatible"]:
            return observations, {
                "mode": "compatible_multi_geometry_fusion",
                "pair": pair,
                "pointer_consistency": pointer,
                "discarded": [],
            }

        first_pointer = pointer[first.name].get("probability")
        second_pointer = pointer[second.name].get("probability")
        pepd_strong = bool(
            pepd.available
            and pepd.confidence >= 0.35
            and math.degrees(pepd.angle_std_radians)
            <= self.config.strong_pepd_std_degrees
        )
        if pepd_strong and first_pointer is not None and second_pointer is not None:
            first_wins = bool(
                first_pointer >= self.config.pointer_branch_inside_probability
                and second_pointer <= self.config.pointer_branch_outside_probability
            )
            second_wins = bool(
                second_pointer >= self.config.pointer_branch_inside_probability
                and first_pointer <= self.config.pointer_branch_outside_probability
            )
            if first_wins != second_wins:
                selected, discarded = (
                    (first, second) if first_wins else (second, first)
                )
                return [selected], {
                    "mode": "pepd_resolved_arc_branch",
                    "pair": pair,
                    "pointer_consistency": pointer,
                    "discarded": [discarded.name],
                }

        first_strong = first.confidence >= self.config.strong_geometry_confidence
        second_strong = second.confidence >= self.config.strong_geometry_confidence
        if first_strong != second_strong:
            selected, discarded = (
                (first, second) if first_strong else (second, first)
            )
            return [selected], {
                "mode": "weak_conflicting_geometry_discarded",
                "pair": pair,
                "pointer_consistency": pointer,
                "discarded": [discarded.name],
            }
        telemetry = {
            "mode": "rejected_geometry_conflict",
            "pair": pair,
            "pointer_consistency": pointer,
            "discarded": [],
        }
        raise GeometryFusionRejected(
            "unresolved_strong_geometry_branch_conflict",
            telemetry=telemetry,
        )

    def _fuse(
        self,
        selected: list[_GeometryObservation],
        pepd: _PEPDObservation,
        selection: Mapping[str, Any],
    ) -> tuple[GeometryHint, dict[str, Any]]:
        pivot_inputs = [
            (
                np.asarray(observation.hint.pivot_xy, dtype=np.float64),
                observation.pivot_covariance,
                max(observation.confidence, 1e-3),
            )
            for observation in selected
        ]
        geometry_pivot, geometry_pivot_covariance = _precision_fuse(pivot_inputs)
        pepd_pivot_used = False
        pepd_pivot_conflict = False
        if pepd.pivot_xy is not None and pepd.pivot_covariance is not None:
            delta = pepd.pivot_xy - geometry_pivot
            combined = pepd.pivot_covariance + geometry_pivot_covariance
            mahalanobis = float(
                math.sqrt(max(0.0, float(delta @ np.linalg.pinv(combined) @ delta)))
            )
            distance = float(np.linalg.norm(delta))
            pepd_pivot_conflict = bool(
                distance > self.config.maximum_pivot_distance_fraction
                and mahalanobis > self.config.maximum_pivot_mahalanobis
            )
            if pepd_pivot_conflict:
                if (
                    pepd.available
                    and pepd.confidence >= 0.50
                    and any(
                        item.confidence >= self.config.strong_geometry_confidence
                        for item in selected
                    )
                ):
                    raise GeometryFusionRejected("strong_pepd_geometry_pivot_conflict")
            else:
                geometry_precision = np.linalg.pinv(geometry_pivot_covariance)
                pepd_precision = max(pepd.confidence, 1e-3) * np.linalg.pinv(
                    pepd.pivot_covariance
                )
                trace_limit = (
                    self.config.pepd_pivot_precision_cap_fraction
                    * float(np.trace(geometry_precision))
                )
                trace = float(np.trace(pepd_precision))
                if trace > trace_limit > 0.0:
                    pepd_precision *= trace_limit / trace
                total = geometry_precision + pepd_precision
                geometry_pivot_covariance = np.linalg.pinv(total)
                geometry_pivot = geometry_pivot_covariance @ (
                    geometry_precision @ geometry_pivot
                    + pepd_precision @ pepd.pivot_xy
                )
                pepd_pivot_used = True

        start_weights = [
            _geometry_weight(item.confidence, item.start_std_radians)
            for item in selected
        ]
        end_weights = [
            _geometry_weight(item.confidence, item.end_std_radians)
            for item in selected
        ]
        start_angle = _circular_mean(
            [item.start_angle for item in selected], start_weights
        )
        end_angle = _circular_mean(
            [item.end_angle for item in selected], end_weights
        )
        start_radius = float(
            np.average([item.start_radius for item in selected], weights=start_weights)
        )
        end_radius = float(
            np.average([item.end_radius for item in selected], weights=end_weights)
        )
        start_xy = _bounded_endpoint(geometry_pivot, start_angle, start_radius)
        end_xy = _bounded_endpoint(geometry_pivot, end_angle, end_radius)

        arc = _directed_arc(start_angle, end_angle)
        if not math.radians(10.0) < arc < math.radians(350.0):
            raise GeometryFusionRejected("fused_arc_outside_valid_bounds")
        fused_pointer_outside = 0.0
        pointer_probability = 1.0
        if pepd.available and pepd.pointer_angle is not None:
            delta = (pepd.pointer_angle - start_angle) % _TAU
            if delta > arc:
                fused_pointer_outside = min(
                    _circular_distance(pepd.pointer_angle, start_angle),
                    _circular_distance(pepd.pointer_angle, end_angle),
                )
                pointer_probability = math.exp(
                    -0.5
                    * (
                        fused_pointer_outside
                        / max(pepd.angle_std_radians, math.radians(1.0))
                    )
                    ** 2
                )
            confident_limit = max(
                math.radians(self.config.maximum_confident_pointer_outside_degrees),
                3.0 * pepd.angle_std_radians,
            )
            if (
                math.degrees(pepd.angle_std_radians)
                <= self.config.strong_pepd_std_degrees
                and fused_pointer_outside > confident_limit
            ):
                raise GeometryFusionRejected("confident_pepd_pointer_outside_fused_arc")

        confidence_weights = [
            1.0 / max(item.start_std_radians * item.end_std_radians, 1e-6)
            for item in selected
        ]
        base_confidence = float(
            np.average(
                [item.confidence for item in selected],
                weights=confidence_weights,
            )
        )
        if len(selected) == 1:
            base_confidence *= self.config.single_geometry_confidence_penalty
        pair = selection.get("pair")
        agreement_penalty = 1.0
        if isinstance(pair, Mapping) and pair.get("compatible") is True:
            endpoint_ratio = max(
                float(pair["start_disagreement_degrees"]),
                float(pair["end_disagreement_degrees"]),
            ) / self.config.maximum_endpoint_disagreement_degrees
            pivot_ratio = float(pair["pivot_distance_fraction"]) / self.config.maximum_pivot_distance_fraction
            agreement_penalty = math.exp(-0.25 * (endpoint_ratio**2 + pivot_ratio**2))
        base_confidence *= agreement_penalty * pointer_probability
        if pepd.available:
            base_confidence *= 0.85 + 0.15 * pepd.confidence
        else:
            base_confidence *= self.config.missing_pepd_confidence_penalty
        confidence = _clip01(base_confidence)
        hint = GeometryHint(
            pivot_xy=(float(geometry_pivot[0]), float(geometry_pivot[1])),
            start_xy=start_xy,
            end_xy=end_xy,
            confidence=confidence,
            source=f"{PROTOCOL}:{'+'.join(item.name for item in selected)}",
        ).validate()
        telemetry = {
            "pepd_pivot_used": pepd_pivot_used,
            "pepd_pivot_conflict": pepd_pivot_conflict,
            "fused_pointer_outside_degrees": math.degrees(fused_pointer_outside),
            "pointer_consistency_probability": pointer_probability,
            "agreement_penalty": agreement_penalty,
            "base_component_confidence": float(
                np.average(
                    [item.confidence for item in selected],
                    weights=confidence_weights,
                )
            ),
            "fused_pivot_covariance_normalized": geometry_pivot_covariance.tolist(),
            "fused_start_angle_degrees": math.degrees(start_angle),
            "fused_end_angle_degrees": math.degrees(end_angle),
            "fused_arc_degrees": math.degrees(arc),
        }
        return hint, telemetry

    def predict(self, image_bgr: np.ndarray) -> GeometryProviderResult:
        """Fuse internal image-derived evidence; no other arguments are accepted."""

        image = validate_canonical_roi(image_bgr)
        input_digest = image_sha256(image)
        with self._lock:
            observations: list[_GeometryObservation] = []
            failures: dict[str, str] = {}
            v5, failure = self._call_geometry(
                "v5", self.v5_geometry_provider, image, input_digest
            )
            if v5 is None:
                failures["v5"] = str(failure)
            else:
                observations.append(v5)
            if self.mask_geometry_provider is not None:
                mask, failure = self._call_geometry(
                    "mask_geometry",
                    self.mask_geometry_provider,
                    image,
                    input_digest,
                )
                if mask is None:
                    failures["mask_geometry"] = str(failure)
                else:
                    observations.append(mask)
            pepd = self._call_pepd(image, input_digest)

            base_telemetry = {
                "protocol": PROTOCOL,
                "input_attestation": {
                    "input_image_sha256": input_digest,
                    "one_whole_canonical_roi": True,
                    "component_inputs_are_copies": True,
                    "caller_points_consumed": False,
                    "caller_reference_consumed": False,
                    "physical_numeric_range_consumed": False,
                },
                "coordinate_frame": COORDINATE_FRAME,
                "component_failures": failures,
                "available_geometry_sources": [item.name for item in observations],
                "pepd": {
                    "available": pepd.available,
                    "failure_code": pepd.failure_code,
                    **dict(pepd.telemetry),
                },
            }
            if not observations:
                raise GeometryFusionRejected(
                    "no_valid_automatic_geometry",
                    telemetry=base_telemetry,
                )
            try:
                selected, selection = self._select_geometry(observations, pepd)
                hint, fused = self._fuse(selected, pepd, selection)
            except GeometryFusionRejected as exc:
                exc.telemetry = {
                    **base_telemetry,
                    **dict(exc.telemetry),
                    "status": "rejected",
                    "failure_code": exc.code,
                }
                raise
            telemetry = {
                **base_telemetry,
                "status": "accepted",
                "selection": selection,
                "selected_geometry_sources": [item.name for item in selected],
                "candidate_summary": {
                    item.name: {
                        "confidence": item.confidence,
                        "pivot_covariance_normalized": item.pivot_covariance.tolist(),
                        "start_angle_std_degrees": math.degrees(item.start_std_radians),
                        "end_angle_std_degrees": math.degrees(item.end_std_radians),
                        "arc_degrees": math.degrees(item.arc),
                    }
                    for item in observations
                },
                "fusion": fused,
            }
            reject_supervised_fields(telemetry, path="garc_geometry_fusion_output")
            return GeometryProviderResult(hint=hint, telemetry=telemetry)


__all__ = [
    "COORDINATE_FRAME",
    "GARCCovarianceGeometryFusionProvider",
    "GeometryFusionConfig",
    "GeometryFusionRejected",
    "ImageOnlyPEPDStructuralProvider",
    "PROTOCOL",
]
