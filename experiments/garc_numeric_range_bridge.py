"""Label-free bridge from automatic OCR/geometry to GARC range consensus.

The public inference boundary is intentionally one canonical BGR meter ROI.
Geometry, OCR boxes, top-K text posteriors, box-to-arc progress and optional
tick proximity are all produced inside the provider.  Caller-supplied boxes,
geometry and physical scale values are not part of the API.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any, Final, Mapping, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np

from experiments.automatic_numeric_range import (
    AutomaticGeometryProvider,
    GeometryHint,
    GeometryProviderResult,
    NumericRangePrediction,
    directed_progress,
    image_sha256,
    validate_canonical_roi,
)
from experiments.garc_posterior_consensus import (
    ArcPosteriorToken,
    GARCConsensusConfig,
    GARCPosteriorConsensusDecoder,
    NumericPosteriorCandidate,
)
from experiments.syncg_numeric_ocr import OCRPosteriorToken


PROTOCOL: Final[str] = "garc_label_free_numeric_range_bridge_v1"


@runtime_checkable
class PosteriorOCRBackend(Protocol):
    """The posterior interface implemented by ``GaugeNumericOCRBackend``."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def infer_with_posteriors(
        self, image_bgr: np.ndarray
    ) -> tuple[list[Any], list[OCRPosteriorToken], float]: ...


@runtime_checkable
class AutomaticTickProximityProvider(Protocol):
    """Optional image-only tick alignment provider configured internally."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def score(
        self,
        image_bgr: np.ndarray,
        box: tuple[tuple[float, float], ...],
        geometry: GeometryHint,
    ) -> float: ...


def _box_trace_base(
    source_index: int,
    token: OCRPosteriorToken,
) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "box": [[float(x), float(y)] for x, y in token.box],
        "detector_score": float(token.detector_score),
        "posterior_candidates": [
            {
                "text": str(candidate.text),
                "log_posterior": float(candidate.log_probability),
                "retained_beam_probability": float(candidate.beam_probability),
            }
            for candidate in token.hypotheses
        ],
    }


def posterior_tokens_to_arc_tokens(
    posterior_tokens: Sequence[OCRPosteriorToken],
    *,
    image_bgr: np.ndarray,
    geometry: GeometryHint,
    tick_proximity_provider: AutomaticTickProximityProvider | None = None,
    annulus_inner_ratio: float = 0.52,
    annulus_outer_ratio: float = 1.18,
    endpoint_margin_fraction: float = 0.10,
) -> tuple[tuple[ArcPosteriorToken, ...], dict[str, Any]]:
    """Convert every predicted OCR box to an automatic arc-progress trace.

    Rejected boxes remain in ``box_trace`` with their automatically calculated
    progress (when defined), original top-K posterior and an explicit reason.
    Thus low coverage can be diagnosed without opening any range label.
    """

    image = validate_canonical_roi(image_bgr)
    geometry.validate()
    if not 0.0 <= annulus_inner_ratio < annulus_outer_ratio:
        raise ValueError("invalid numeric annulus bounds")
    if not 0.0 <= endpoint_margin_fraction <= 0.25:
        raise ValueError("endpoint_margin_fraction must be in [0,0.25]")
    if tick_proximity_provider is not None and not isinstance(
        tick_proximity_provider, AutomaticTickProximityProvider
    ):
        raise TypeError("tick provider does not implement AutomaticTickProximityProvider")

    height, width = image.shape[:2]
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
        raise ValueError("automatic geometry reference radius is too small")

    counts = {
        "posterior_boxes": len(posterior_tokens),
        "accepted_boxes": 0,
        "invalid_box": 0,
        "oversized_box": 0,
        "outside_annulus": 0,
        "outside_arc": 0,
        "no_valid_numeric_posterior": 0,
        "invalid_tick_proximity": 0,
    }
    arc_tokens: list[ArcPosteriorToken] = []
    traces: list[dict[str, Any]] = []
    for source_index, token in enumerate(posterior_tokens):
        if not isinstance(token, OCRPosteriorToken):
            raise TypeError("posterior backend returned a non-OCRPosteriorToken")
        trace = _box_trace_base(source_index, token)
        box = np.asarray(token.box, dtype=np.float64)
        if box.shape != (4, 2) or not np.isfinite(box).all():
            counts["invalid_box"] += 1
            trace.update(
                {
                    "center_xy": None,
                    "automatic_arc_progress": None,
                    "radius_ratio": None,
                    "accepted": False,
                    "failure_reason": "invalid_predicted_text_box",
                }
            )
            traces.append(trace)
            continue

        center = box.mean(axis=0)
        center_normalized = (
            float(center[0] / width),
            float(center[1] / height),
        )
        progress = directed_progress(
            center_normalized,
            geometry,
            endpoint_margin_fraction=endpoint_margin_fraction,
        )
        radius_ratio = float(np.linalg.norm(center - pivot) / reference_radius)
        trace.update(
            {
                "center_xy": [float(center[0]), float(center[1])],
                "center_normalized_xy": list(center_normalized),
                "automatic_arc_progress": (
                    None if progress is None else float(progress)
                ),
                "radius_ratio": radius_ratio,
            }
        )

        box_width = float(np.ptp(box[:, 0]))
        box_height = float(np.ptp(box[:, 1]))
        if box_width > 0.28 * width or box_height > 0.18 * height:
            counts["oversized_box"] += 1
            trace.update(
                {"accepted": False, "failure_reason": "predicted_text_box_too_large"}
            )
            traces.append(trace)
            continue
        if not annulus_inner_ratio <= radius_ratio <= annulus_outer_ratio:
            counts["outside_annulus"] += 1
            trace.update(
                {"accepted": False, "failure_reason": "box_outside_predicted_annulus"}
            )
            traces.append(trace)
            continue
        if progress is None or not (
            -endpoint_margin_fraction
            <= progress
            <= 1.0 + endpoint_margin_fraction
        ):
            counts["outside_arc"] += 1
            trace.update(
                {"accepted": False, "failure_reason": "box_outside_predicted_arc"}
            )
            traces.append(trace)
            continue

        candidates: list[NumericPosteriorCandidate] = []
        rejected_candidates: list[dict[str, Any]] = []
        for candidate_index, hypothesis in enumerate(token.hypotheses):
            try:
                candidate = NumericPosteriorCandidate(
                    text=str(hypothesis.text),
                    log_posterior=float(hypothesis.log_probability),
                ).validate()
            except (TypeError, ValueError) as error:
                rejected_candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "reason": str(error),
                    }
                )
                continue
            candidates.append(candidate)
        trace["rejected_posterior_candidates"] = rejected_candidates
        if not candidates:
            counts["no_valid_numeric_posterior"] += 1
            trace.update(
                {"accepted": False, "failure_reason": "no_valid_numeric_topk_candidate"}
            )
            traces.append(trace)
            continue

        tick_proximity: float | None = None
        if tick_proximity_provider is not None:
            try:
                tick_proximity = float(
                    tick_proximity_provider.score(image, token.box, geometry)
                )
            except (TypeError, ValueError, ArithmeticError) as error:
                counts["invalid_tick_proximity"] += 1
                trace.update(
                    {
                        "accepted": False,
                        "failure_reason": "tick_proximity_provider_failed",
                        "tick_proximity_error": str(error),
                    }
                )
                traces.append(trace)
                continue
            if not math.isfinite(tick_proximity) or not 0.0 <= tick_proximity <= 1.0:
                counts["invalid_tick_proximity"] += 1
                trace.update(
                    {
                        "accepted": False,
                        "failure_reason": "invalid_automatic_tick_proximity",
                        "tick_proximity": tick_proximity,
                    }
                )
                traces.append(trace)
                continue

        clipped_progress = float(np.clip(progress, 0.0, 1.0))
        arc_token = ArcPosteriorToken(
            source_index=source_index,
            progress=clipped_progress,
            candidates=tuple(candidates),
            tick_proximity=tick_proximity,
        ).validate()
        arc_tokens.append(arc_token)
        counts["accepted_boxes"] += 1
        trace.update(
            {
                "accepted": True,
                "failure_reason": None,
                "consensus_progress": clipped_progress,
                "tick_proximity": tick_proximity,
                "accepted_candidate_count": len(candidates),
            }
        )
        traces.append(trace)

    telemetry = {
        "conversion": "predicted_box_center_to_predicted_directed_arc_progress",
        "manual_box_input": False,
        "manual_geometry_input": False,
        "known_range_input": False,
        "geometry": asdict(geometry),
        "annulus_inner_ratio": annulus_inner_ratio,
        "annulus_outer_ratio": annulus_outer_ratio,
        "endpoint_margin_fraction": endpoint_margin_fraction,
        "filter_counts": counts,
        "box_trace": traces,
    }
    return tuple(arc_tokens), telemetry


class GARCAutomaticNumericRangeProvider:
    """One-ROI-in, automatic real-valued scale-range-out provider."""

    def __init__(
        self,
        geometry_provider: AutomaticGeometryProvider,
        posterior_ocr_backend: PosteriorOCRBackend,
        *,
        input_size: int = 768,
        consensus_config: GARCConsensusConfig | None = None,
        tick_proximity_provider: AutomaticTickProximityProvider | None = None,
    ) -> None:
        if not isinstance(geometry_provider, AutomaticGeometryProvider):
            raise TypeError("geometry provider does not implement AutomaticGeometryProvider")
        if not isinstance(posterior_ocr_backend, PosteriorOCRBackend):
            raise TypeError("OCR backend does not implement PosteriorOCRBackend")
        if int(input_size) not in (512, 768):
            raise ValueError("GARC OCR input_size must be 512 or 768")
        if tick_proximity_provider is not None and not isinstance(
            tick_proximity_provider, AutomaticTickProximityProvider
        ):
            raise TypeError("tick provider does not implement AutomaticTickProximityProvider")
        self.geometry_provider = geometry_provider
        self.posterior_ocr_backend = posterior_ocr_backend
        self.input_size = int(input_size)
        self.tick_proximity_provider = tick_proximity_provider
        self.decoder = GARCPosteriorConsensusDecoder(consensus_config)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "protocol": PROTOCOL,
            "primary_input": "one whole canonical meter ROI (BGR uint8)",
            "caller_supplied_boxes_allowed": False,
            "caller_supplied_geometry_allowed": False,
            "caller_supplied_range_allowed": False,
            "geometry_provider": dict(self.geometry_provider.identity),
            "posterior_ocr_backend": dict(self.posterior_ocr_backend.identity),
            "tick_proximity_provider": (
                None
                if self.tick_proximity_provider is None
                else dict(self.tick_proximity_provider.identity)
            ),
            "ocr_input_size": self.input_size,
            "consensus": dict(self.decoder.identity),
        }

    def predict(
        self, canonical_meter_roi_bgr: np.ndarray
    ) -> NumericRangePrediction:
        started = time.perf_counter()
        image = validate_canonical_roi(canonical_meter_roi_bgr)
        input_digest = image_sha256(image)
        provided = self.geometry_provider.predict(image)
        if not isinstance(provided, GeometryProviderResult):
            raise TypeError("automatic geometry provider returned an invalid result")
        geometry = provided.hint.validate()

        interpolation = (
            cv2.INTER_AREA
            if max(image.shape[:2]) > self.input_size
            else cv2.INTER_CUBIC
        )
        ocr_image = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=interpolation,
        )
        backend_output = self.posterior_ocr_backend.infer_with_posteriors(ocr_image)
        if not isinstance(backend_output, tuple) or len(backend_output) != 3:
            raise TypeError("posterior OCR backend returned an invalid result")
        top1_tokens, posterior_tokens, ocr_seconds = backend_output
        if not isinstance(posterior_tokens, list):
            raise TypeError("posterior OCR token collection must be a list")
        if not math.isfinite(float(ocr_seconds)) or float(ocr_seconds) < 0.0:
            raise ValueError("posterior OCR runtime must be finite and non-negative")

        arc_tokens, bridge_telemetry = posterior_tokens_to_arc_tokens(
            posterior_tokens,
            image_bgr=ocr_image,
            geometry=geometry,
            tick_proximity_provider=self.tick_proximity_provider,
        )
        consensus = self.decoder.predict(arc_tokens)
        telemetry = {
            "primary_adapter": {
                "accepts_only_one_whole_roi": True,
                "accepts_manual_boxes": False,
                "accepts_manual_geometry": False,
                "accepts_physical_scale_values": False,
                "input_image_sha256": input_digest,
                "geometry_and_ocr_same_source_image_sha256": input_digest,
            },
            "whole_roi_direct_resize": True,
            "second_caller_crop": False,
            "ocr_input_shape": list(ocr_image.shape),
            "ocr_seconds": float(ocr_seconds),
            "total_seconds": float(time.perf_counter() - started),
            "top1_compatibility_token_count": len(top1_tokens),
            "posterior_token_count": len(posterior_tokens),
            "geometry_provider": dict(self.geometry_provider.identity),
            "geometry_telemetry": dict(provided.telemetry),
            "posterior_ocr_backend": dict(self.posterior_ocr_backend.identity),
            "bridge": bridge_telemetry,
            "consensus": consensus.as_dict(),
        }
        return NumericRangePrediction(
            protocol=PROTOCOL,
            status=consensus.status,
            prediction_space=consensus.prediction_space,
            pred_start=consensus.pred_start,
            pred_end=consensus.pred_end,
            confidence=consensus.confidence,
            failure_reason=consensus.failure_reason,
            telemetry=telemetry,
        )
