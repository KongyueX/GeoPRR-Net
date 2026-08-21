"""Posterior arithmetic-range consensus for label-free gauge reading.

This module is the numerical core of the working GARC design.  It consumes
only outputs available at inference time: an automatically computed arc
progress for each OCR box, the OCR recognizer's top-K string posterior, and an
optional tick-proximity score.  It never accepts a scale label or a known
range.

For a range ``(s, e)`` and arc progress ``p_i``, the predicted value at OCR
box ``i`` is

    y_i = s + p_i * (e - s).

Candidate pairs generate deterministic positive-range hypotheses.  Each
hypothesis is scored by marginalizing the robust geometric likelihood over
the complete OCR posterior rather than committing to top-1 text first::

    L(s, e) = sum_i w_i log sum_j q_ij
              exp(max(-0.5 * ((v_ij - y_i) / tau)^2, ell_out)).

The winning assignments are refit with weighted least squares.  A second,
well-supported but materially different range with near-equal evidence causes
a fail-closed ambiguity result.  Outputs remain real-valued: there is no
integer class inventory, endpoint lookup table, or snapping to labelled test
ranges.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np

from experiments.automatic_numeric_range import parse_numeric_text


PROTOCOL: Final[str] = "garc_topk_posterior_arithmetic_consensus_v1"
TOP1_PROTOCOL: Final[str] = "garc_top1_arithmetic_ransac_control_v1"
PREDICTION_SPACE: Final[str] = "real_numeric_scale_start_end"


@dataclass(frozen=True)
class NumericPosteriorCandidate:
    """One OCR string candidate and its unnormalised log posterior."""

    text: str
    log_posterior: float

    def validate(self) -> "NumericPosteriorCandidate":
        if not str(self.text).strip():
            raise ValueError("candidate text must be non-empty")
        if parse_numeric_text(self.text) is None:
            raise ValueError(f"candidate is not a signed real number: {self.text!r}")
        if not math.isfinite(float(self.log_posterior)):
            raise ValueError("candidate log_posterior must be finite")
        return self


@dataclass(frozen=True)
class ArcPosteriorToken:
    """Top-K OCR evidence for one box positioned on an automatic dial arc.

    ``progress`` must be produced from predicted geometry, not from a scale
    annotation.  ``tick_proximity`` is an optional [0, 1] confidence supplied
    by a tick detector; omitting it is neutral.
    """

    source_index: int
    progress: float
    candidates: tuple[NumericPosteriorCandidate, ...]
    tick_proximity: float | None = None

    def validate(self, *, maximum_candidates: int = 10) -> "ArcPosteriorToken":
        if int(self.source_index) != self.source_index or int(self.source_index) < 0:
            raise ValueError("source_index must be a non-negative integer")
        if not math.isfinite(float(self.progress)) or not 0.0 <= self.progress <= 1.0:
            raise ValueError("automatic arc progress must be finite and in [0,1]")
        if not self.candidates:
            raise ValueError("each OCR box must contain at least one numeric candidate")
        if len(self.candidates) > int(maximum_candidates):
            raise ValueError(
                f"OCR top-K exceeds the configured maximum of {maximum_candidates}"
            )
        for candidate in self.candidates:
            if not isinstance(candidate, NumericPosteriorCandidate):
                raise TypeError("candidates must be NumericPosteriorCandidate objects")
            candidate.validate()
        if self.tick_proximity is not None and (
            not math.isfinite(float(self.tick_proximity))
            or not 0.0 <= float(self.tick_proximity) <= 1.0
        ):
            raise ValueError("tick_proximity must be finite and in [0,1]")
        return self


@dataclass(frozen=True)
class GARCConsensusConfig:
    """Frozen decoder hyperparameters; values do not depend on test labels."""

    minimum_inliers: int = 3
    minimum_progress_span: float = 0.18
    minimum_pair_progress_span: float = 0.06
    maximum_abs_range: float = 100_000.0
    maximum_candidates_per_token: int = 10
    maximum_retained_solutions: int = 2_048
    residual_relative_to_range: float = 0.0125
    residual_quantum_multiplier: float = 0.35
    robust_log_likelihood_floor: float = -6.0
    solution_merge_relative_to_range: float = 0.02
    ambiguity_endpoint_relative_to_range: float = 0.05
    minimum_evidence_margin_per_weight: float = 0.12
    minimum_confidence: float = 0.55
    refinement_iterations: int = 4

    def validate(self) -> "GARCConsensusConfig":
        if self.minimum_inliers < 3:
            raise ValueError("minimum_inliers must be at least three")
        if not 0.0 < self.minimum_progress_span <= 1.0:
            raise ValueError("minimum_progress_span must be in (0,1]")
        if not 0.0 < self.minimum_pair_progress_span <= 1.0:
            raise ValueError("minimum_pair_progress_span must be in (0,1]")
        if not math.isfinite(self.maximum_abs_range) or self.maximum_abs_range <= 0.0:
            raise ValueError("maximum_abs_range must be positive and finite")
        if self.maximum_candidates_per_token < 1:
            raise ValueError("maximum_candidates_per_token must be positive")
        if self.maximum_retained_solutions < 2:
            raise ValueError("maximum_retained_solutions must be at least two")
        for name in (
            "residual_relative_to_range",
            "residual_quantum_multiplier",
            "solution_merge_relative_to_range",
            "ambiguity_endpoint_relative_to_range",
            "minimum_evidence_margin_per_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(self.robust_log_likelihood_floor)
            or self.robust_log_likelihood_floor >= 0.0
        ):
            raise ValueError("robust_log_likelihood_floor must be finite and negative")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0,1]")
        if self.refinement_iterations < 1:
            raise ValueError("refinement_iterations must be positive")
        if (
            self.solution_merge_relative_to_range
            >= self.ambiguity_endpoint_relative_to_range
        ):
            raise ValueError(
                "solution merge radius must be smaller than ambiguity separation"
            )
        return self


@dataclass(frozen=True)
class SelectedNumericCandidate:
    source_index: int
    progress: float
    text: str
    value: float
    posterior_probability: float
    tick_weight: float
    residual: float


@dataclass(frozen=True)
class GARCConsensusPrediction:
    protocol: str
    status: bool
    prediction_space: str
    pred_start: float | None
    pred_end: float | None
    confidence: float
    ambiguity: bool
    failure_reason: str | None
    selected_candidates: tuple[SelectedNumericCandidate, ...]
    telemetry: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Candidate:
    text: str
    value: float
    log_probability: float
    probability: float
    original_rank: int


@dataclass(frozen=True)
class _Token:
    source_index: int
    progress: float
    candidates: tuple[_Candidate, ...]
    tick_weight: float


@dataclass(frozen=True)
class _Assignment:
    token: _Token
    candidate: _Candidate
    residual: float
    normalized_residual: float


@dataclass(frozen=True)
class _Solution:
    start: float
    end: float
    tolerance: float
    evidence: float
    evidence_per_weight: float
    inlier_weight: float
    inlier_count: int
    progress_span: float
    normalized_rmse: float
    assignments: tuple[_Assignment, ...]


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return float(maximum + math.log(sum(math.exp(value - maximum) for value in values)))


def _decimal_quantum(text: str) -> float:
    normalized = "".join(str(text).strip().replace("−", "-").split())
    if "," in normalized and "." not in normalized:
        normalized = normalized.replace(",", ".")
    places = len(normalized.rsplit(".", 1)[1]) if "." in normalized else 0
    return 10.0 ** (-min(places, 4))


def _prepare_tokens(
    raw_tokens: Sequence[ArcPosteriorToken],
    config: GARCConsensusConfig,
    *,
    top1_only: bool,
) -> tuple[_Token, ...]:
    if len(raw_tokens) > 128:
        raise ValueError("at most 128 OCR boxes are accepted per meter")
    seen: set[int] = set()
    prepared: list[_Token] = []
    for raw in raw_tokens:
        if not isinstance(raw, ArcPosteriorToken):
            raise TypeError("tokens must be ArcPosteriorToken objects")
        raw.validate(maximum_candidates=config.maximum_candidates_per_token)
        source_index = int(raw.source_index)
        if source_index in seen:
            raise ValueError(f"duplicate source_index: {source_index}")
        seen.add(source_index)

        parsed: list[tuple[int, str, float, float]] = []
        for rank, candidate in enumerate(raw.candidates):
            value = parse_numeric_text(candidate.text)
            assert value is not None  # validated above
            parsed.append((rank, str(candidate.text), float(value), float(candidate.log_posterior)))

        if top1_only:
            # The degradation control means literal recognizer top-1.  It must
            # not benefit from posterior mass aggregation across equivalent
            # spellings, which is part of the proposed top-K decoder.
            rank, text, value, _ = max(
                parsed, key=lambda row: (row[3], -row[0], -row[2], row[1])
            )
            candidates = (
                _Candidate(
                    text=text,
                    value=value,
                    log_probability=0.0,
                    probability=1.0,
                    original_rank=rank,
                ),
            )
        else:
            # OCR heads can emit two spellings for the same number.  Merge
            # their probability mass before fitting so duplicate beams cannot
            # vote twice.
            grouped: dict[float, list[tuple[int, str, float]]] = {}
            for rank, text, value, logit in parsed:
                grouped.setdefault(value, []).append((rank, text, logit))
            merged: list[tuple[int, str, float, float]] = []
            for value, rows in grouped.items():
                merged_logit = _logsumexp([row[2] for row in rows])
                representative = min(rows, key=lambda row: (row[0], row[1]))
                merged.append(
                    (representative[0], representative[1], value, merged_logit)
                )
            merged.sort(key=lambda row: (-row[3], row[2], row[1], row[0]))
            normalizer = _logsumexp([row[3] for row in merged])
            candidates = tuple(
                _Candidate(
                    text=text,
                    value=value,
                    log_probability=logit - normalizer,
                    probability=math.exp(logit - normalizer),
                    original_rank=rank,
                )
                for rank, text, value, logit in merged
            )
        tick_weight = (
            1.0
            if raw.tick_proximity is None
            else 0.5 + 0.5 * float(raw.tick_proximity)
        )
        prepared.append(
            _Token(
                source_index=source_index,
                progress=float(raw.progress),
                candidates=candidates,
                tick_weight=tick_weight,
            )
        )
    prepared.sort(key=lambda token: (token.progress, token.source_index))
    return tuple(prepared)


def _observed_quantum(tokens: Sequence[_Token]) -> float:
    return min(
        _decimal_quantum(candidate.text)
        for token in tokens
        for candidate in token.candidates
    )


def _residual_tolerance(
    value_range: float, quantum: float, config: GARCConsensusConfig
) -> float:
    return max(
        1e-7,
        config.residual_relative_to_range * abs(value_range),
        config.residual_quantum_multiplier * quantum,
    )


def _weighted_line_fit(assignments: Sequence[_Assignment]) -> tuple[float, float]:
    progress = np.asarray([row.token.progress for row in assignments], dtype=np.float64)
    values = np.asarray([row.candidate.value for row in assignments], dtype=np.float64)
    weights = np.asarray(
        [row.token.tick_weight * max(row.candidate.probability, 1e-5) for row in assignments],
        dtype=np.float64,
    )
    design = np.stack((np.ones_like(progress), progress), axis=1)
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_values = values * np.sqrt(weights)
    intercept, slope = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
    return float(intercept), float(slope)


def _evaluate_solution(
    start: float,
    value_range: float,
    tokens: Sequence[_Token],
    *,
    quantum: float,
    config: GARCConsensusConfig,
) -> _Solution:
    tolerance = _residual_tolerance(value_range, quantum, config)
    assignments: list[_Assignment] = []
    evidence = 0.0
    total_weight = sum(token.tick_weight for token in tokens)
    for token in tokens:
        predicted = start + value_range * token.progress
        utilities: list[float] = []
        residuals: list[float] = []
        for candidate in token.candidates:
            residual = abs(candidate.value - predicted)
            normalized = residual / tolerance
            robust_log_likelihood = max(
                -0.5 * normalized * normalized,
                config.robust_log_likelihood_floor,
            )
            utilities.append(candidate.log_probability + robust_log_likelihood)
            residuals.append(residual)
        evidence += token.tick_weight * _logsumexp(utilities)
        best_index = max(
            range(len(token.candidates)),
            key=lambda index: (
                utilities[index],
                token.candidates[index].log_probability,
                -token.candidates[index].value,
                -token.candidates[index].original_rank,
            ),
        )
        residual = residuals[best_index]
        if residual <= tolerance:
            assignments.append(
                _Assignment(
                    token=token,
                    candidate=token.candidates[best_index],
                    residual=residual,
                    normalized_residual=residual / tolerance,
                )
            )

    if assignments:
        inlier_weight = sum(row.token.tick_weight for row in assignments)
        progress_values = [row.token.progress for row in assignments]
        progress_span = max(progress_values) - min(progress_values)
        squared = sum(
            row.token.tick_weight * row.normalized_residual**2 for row in assignments
        )
        normalized_rmse = math.sqrt(squared / inlier_weight)
    else:
        inlier_weight = 0.0
        progress_span = 0.0
        normalized_rmse = math.inf
    return _Solution(
        start=float(start),
        end=float(start + value_range),
        tolerance=float(tolerance),
        evidence=float(evidence),
        evidence_per_weight=float(evidence / max(total_weight, 1e-12)),
        inlier_weight=float(inlier_weight),
        inlier_count=len(assignments),
        progress_span=float(progress_span),
        normalized_rmse=float(normalized_rmse),
        assignments=tuple(assignments),
    )


def _refine_solution(
    start: float,
    value_range: float,
    tokens: Sequence[_Token],
    *,
    quantum: float,
    config: GARCConsensusConfig,
) -> _Solution:
    current = _evaluate_solution(
        start, value_range, tokens, quantum=quantum, config=config
    )
    previous_signature: tuple[tuple[int, float], ...] | None = None
    for _ in range(config.refinement_iterations):
        if len(current.assignments) < 2:
            break
        signature = tuple(
            (row.token.source_index, row.candidate.value) for row in current.assignments
        )
        if signature == previous_signature:
            break
        previous_signature = signature
        intercept, slope = _weighted_line_fit(current.assignments)
        if not 0.0 < slope <= config.maximum_abs_range:
            break
        current = _evaluate_solution(
            intercept, slope, tokens, quantum=quantum, config=config
        )
    return current


def _solution_order_key(solution: _Solution) -> tuple[float, ...]:
    # Evidence is the primary posterior objective.  Remaining entries make
    # exact ties deterministic without introducing a scale-class prior.
    return (
        solution.evidence,
        float(solution.inlier_count),
        solution.inlier_weight,
        solution.progress_span,
        -solution.normalized_rmse,
        -abs(solution.end - solution.start),
        -solution.start,
        -solution.end,
    )


def _is_eligible(solution: _Solution, config: GARCConsensusConfig) -> bool:
    return (
        solution.inlier_count >= config.minimum_inliers
        and solution.progress_span >= config.minimum_progress_span
        and 0.0 < solution.end - solution.start <= config.maximum_abs_range
    )


def _solution_distance(left: _Solution, right: _Solution) -> float:
    scale = max(
        abs(left.end - left.start),
        abs(right.end - right.start),
        left.tolerance,
        right.tolerance,
        1e-7,
    )
    return max(abs(left.start - right.start), abs(left.end - right.end)) / scale


def _deduplicate_solutions(
    solutions: Sequence[_Solution], config: GARCConsensusConfig
) -> list[_Solution]:
    distinct: list[_Solution] = []
    ranked = sorted(solutions, key=_solution_order_key, reverse=True)[
        : config.maximum_retained_solutions
    ]
    for solution in ranked:
        if any(
            _solution_distance(solution, existing)
            <= config.solution_merge_relative_to_range
            for existing in distinct
        ):
            continue
        distinct.append(solution)
    return distinct


def _selected_public(assignments: Sequence[_Assignment]) -> tuple[SelectedNumericCandidate, ...]:
    return tuple(
        SelectedNumericCandidate(
            source_index=row.token.source_index,
            progress=row.token.progress,
            text=row.candidate.text,
            value=row.candidate.value,
            posterior_probability=row.candidate.probability,
            tick_weight=row.token.tick_weight,
            residual=row.residual,
        )
        for row in assignments
    )


def _base_telemetry(
    tokens: Sequence[_Token], config: GARCConsensusConfig, *, top1_only: bool
) -> dict[str, Any]:
    return {
        "decoder": (
            "top1_pair_ransac_control"
            if top1_only
            else "topk_posterior_marginalized_arithmetic_consensus"
        ),
        "label_or_known_range_input": False,
        "candidate_space": "signed_real_numeric_strings",
        "integer_range_classification": False,
        "endpoint_snapping": False,
        "automatic_arc_progress_required": True,
        "objective": (
            "sum_i tick_weight_i * logsumexp_j(log_q_ij + "
            "max(-0.5*(residual_ij/tau)^2, robust_floor))"
        ),
        "config": asdict(config),
        "input_token_count": len(tokens),
        "input_candidate_count": sum(len(token.candidates) for token in tokens),
    }


def _failure(
    *,
    protocol: str,
    reason: str,
    telemetry: Mapping[str, Any],
    ambiguity: bool = False,
    confidence: float = 0.0,
) -> GARCConsensusPrediction:
    return GARCConsensusPrediction(
        protocol=protocol,
        status=False,
        prediction_space=PREDICTION_SPACE,
        pred_start=None,
        pred_end=None,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        ambiguity=bool(ambiguity),
        failure_reason=reason,
        selected_candidates=(),
        telemetry=telemetry,
    )


def _solve(
    raw_tokens: Sequence[ArcPosteriorToken],
    *,
    config: GARCConsensusConfig,
    top1_only: bool,
) -> GARCConsensusPrediction:
    config.validate()
    protocol = TOP1_PROTOCOL if top1_only else PROTOCOL
    tokens = _prepare_tokens(raw_tokens, config, top1_only=top1_only)
    telemetry = _base_telemetry(tokens, config, top1_only=top1_only)
    if len(tokens) < config.minimum_inliers:
        return _failure(
            protocol=protocol,
            reason="fewer_than_minimum_numeric_tokens",
            telemetry=telemetry,
        )
    observed_progress_span = max(token.progress for token in tokens) - min(
        token.progress for token in tokens
    )
    if observed_progress_span < config.minimum_progress_span:
        return _failure(
            protocol=protocol,
            reason="insufficient_automatic_arc_progress_span",
            telemetry=telemetry,
        )

    quantum = _observed_quantum(tokens)
    solutions: list[_Solution] = []
    hypothesis_count = 0
    for left_index, left in enumerate(tokens):
        for right in tokens[left_index + 1 :]:
            progress_delta = right.progress - left.progress
            if progress_delta < config.minimum_pair_progress_span:
                continue
            for left_candidate in left.candidates:
                for right_candidate in right.candidates:
                    value_range = (
                        right_candidate.value - left_candidate.value
                    ) / progress_delta
                    if not 0.0 < value_range <= config.maximum_abs_range:
                        continue
                    start = left_candidate.value - value_range * left.progress
                    if not math.isfinite(start):
                        continue
                    hypothesis_count += 1
                    solutions.append(
                        _refine_solution(
                            start,
                            value_range,
                            tokens,
                            quantum=quantum,
                            config=config,
                        )
                    )
    telemetry["generated_positive_pair_hypotheses"] = hypothesis_count
    telemetry["retained_hypotheses_for_mode_search"] = min(
        hypothesis_count, config.maximum_retained_solutions
    )
    telemetry["observed_decimal_quantum"] = quantum
    if not solutions:
        return _failure(
            protocol=protocol,
            reason="no_positive_arithmetic_range_hypothesis",
            telemetry=telemetry,
        )

    distinct = _deduplicate_solutions(solutions, config)
    eligible = [solution for solution in distinct if _is_eligible(solution, config)]
    telemetry["distinct_solution_count"] = len(distinct)
    telemetry["eligible_solution_count"] = len(eligible)
    if not eligible:
        best_any = max(distinct, key=_solution_order_key)
        telemetry["best_ineligible"] = {
            "pred_start": best_any.start,
            "pred_end": best_any.end,
            "inlier_count": best_any.inlier_count,
            "progress_span": best_any.progress_span,
            "evidence_per_weight": best_any.evidence_per_weight,
        }
        return _failure(
            protocol=protocol,
            reason="insufficient_arithmetic_consensus",
            telemetry=telemetry,
        )

    eligible.sort(key=_solution_order_key, reverse=True)
    best = eligible[0]
    alternative = next(
        (
            solution
            for solution in eligible[1:]
            if _solution_distance(best, solution)
            >= config.ambiguity_endpoint_relative_to_range
        ),
        None,
    )
    total_weight = sum(token.tick_weight for token in tokens)
    evidence_margin = (
        math.inf
        if alternative is None
        else (best.evidence - alternative.evidence) / max(total_weight, 1e-12)
    )
    telemetry["best_solution"] = {
        "pred_start": best.start,
        "pred_end": best.end,
        "evidence": best.evidence,
        "evidence_per_weight": best.evidence_per_weight,
        "inlier_count": best.inlier_count,
        "inlier_weight": best.inlier_weight,
        "progress_span": best.progress_span,
        "normalized_rmse": best.normalized_rmse,
        "residual_tolerance": best.tolerance,
    }
    telemetry["alternative_solution"] = (
        None
        if alternative is None
        else {
            "pred_start": alternative.start,
            "pred_end": alternative.end,
            "evidence": alternative.evidence,
            "evidence_per_weight": alternative.evidence_per_weight,
            "inlier_count": alternative.inlier_count,
            "progress_span": alternative.progress_span,
            "normalized_rmse": alternative.normalized_rmse,
        }
    )
    telemetry["evidence_margin_per_weight"] = (
        None if math.isinf(evidence_margin) else evidence_margin
    )

    if evidence_margin < config.minimum_evidence_margin_per_weight:
        return _failure(
            protocol=protocol,
            reason="ambiguous_arithmetic_range_hypotheses",
            telemetry=telemetry,
            ambiguity=True,
        )

    support_score = best.inlier_weight / max(total_weight, 1e-12)
    span_score = min(1.0, best.progress_span / max(config.minimum_progress_span * 2.0, 1e-12))
    residual_score = math.exp(-best.normalized_rmse)
    posterior_score = sum(
        row.token.tick_weight * row.candidate.probability for row in best.assignments
    ) / max(best.inlier_weight, 1e-12)
    margin_score = (
        1.0
        if math.isinf(evidence_margin)
        else 1.0 - math.exp(-evidence_margin / config.minimum_evidence_margin_per_weight)
    )
    confidence = float(
        np.clip(
            0.30 * support_score
            + 0.20 * span_score
            + 0.20 * residual_score
            + 0.10 * posterior_score
            + 0.20 * margin_score,
            0.0,
            1.0,
        )
    )
    telemetry["confidence_components"] = {
        "support": support_score,
        "progress_span": span_score,
        "residual": residual_score,
        "selected_ocr_posterior": posterior_score,
        "hypothesis_margin": margin_score,
    }
    if confidence < config.minimum_confidence:
        return _failure(
            protocol=protocol,
            reason="consensus_confidence_below_frozen_threshold",
            telemetry=telemetry,
            confidence=confidence,
        )

    return GARCConsensusPrediction(
        protocol=protocol,
        status=True,
        prediction_space=PREDICTION_SPACE,
        pred_start=best.start,
        pred_end=best.end,
        confidence=confidence,
        ambiguity=False,
        failure_reason=None,
        selected_candidates=_selected_public(best.assignments),
        telemetry=telemetry,
    )


def decode_posterior_numeric_range(
    tokens: Sequence[ArcPosteriorToken],
    *,
    config: GARCConsensusConfig | None = None,
) -> GARCConsensusPrediction:
    """Decode a real-valued range from label-free top-K OCR evidence."""

    return _solve(
        tokens,
        config=(config or GARCConsensusConfig()),
        top1_only=False,
    )


def decode_top1_ransac_control(
    tokens: Sequence[ArcPosteriorToken],
    *,
    config: GARCConsensusConfig | None = None,
) -> GARCConsensusPrediction:
    """A deterministic degradation control that discards all but OCR top-1."""

    return _solve(
        tokens,
        config=(config or GARCConsensusConfig()),
        top1_only=True,
    )


class GARCPosteriorConsensusDecoder:
    """Small inference adapter for a detector/recognizer training backend.

    The upstream backend only needs to convert its box progress and top-K beam
    logits into :class:`ArcPosteriorToken` objects.  No ground-truth or
    model-specific tensor type crosses this boundary.
    """

    def __init__(self, config: GARCConsensusConfig | None = None):
        self.config = (config or GARCConsensusConfig()).validate()

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "protocol": PROTOCOL,
            "prediction_space": PREDICTION_SPACE,
            "input": "automatic arc progress + OCR top-K numeric log posterior",
            "known_range_input": False,
            "ground_truth_input": False,
            "config": asdict(self.config),
        }

    def predict(
        self, tokens: Sequence[ArcPosteriorToken]
    ) -> GARCConsensusPrediction:
        return decode_posterior_numeric_range(tokens, config=self.config)

    def top1_control(
        self, tokens: Sequence[ArcPosteriorToken]
    ) -> GARCConsensusPrediction:
        return decode_top1_ransac_control(tokens, config=self.config)
