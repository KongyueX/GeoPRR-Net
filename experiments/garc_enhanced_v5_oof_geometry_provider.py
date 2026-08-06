"""Fold-authenticated enhanced-V5 geometry provider for GARC.

The historical ``FrozenPublicScaleMarkGeometryProvider`` silently couples a
V5 head to the fixed PEPD seed-20260720 feature extractor.  That is useful for
development sensitivity analyses, but it cannot establish common unseen
end-to-end evidence for rows routed to the other PEPD folds.

This provider binds one enhanced-V5 OOF head to the *same* authoritative PEPD
checkpoint used to train that head.  Its constructor authenticates the final
OOF summary, the per-fold summary, both checkpoints, and the zero group-overlap
attestation before loading any model.  It never opens an image during binding
or construction; ``predict`` accepts only the already canonicalized meter ROI.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

from experiments.automatic_numeric_range import GeometryHint, GeometryProviderResult
from experiments.automatic_numeric_range_public_protocol import (
    guard_public_path,
    require,
    sha256_file,
    strict_json,
)
from experiments.cagh_scalemark_reference_head_v5 import build_head
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import (
    decode_probabilistic_pivot_direction,
)
import experiments.train_cagh_scalemark_reference_probe_v5 as public_v5


OOF_PROTOCOL: Final[str] = "cagh_v5_enhanced_authoritative_pepd_oof_v1"
PROVIDER_PROTOCOL: Final[str] = "garc_enhanced_v5_oof_geometry_provider_v1"
SUPPORTED_PEPD_SEEDS: Final[tuple[int, ...]] = (20260720, 20260721, 20260722)
SOURCE: Final[Path] = Path(__file__).resolve()


def _binding(path: Path, *, label: str) -> dict[str, str]:
    value = guard_public_path(path, label=label)
    return {"path": str(value), "sha256": sha256_file(value)}


def authenticate_oof_fold(
    *,
    oof_summary_path: Path,
    pepd_seed: int,
    head_checkpoint_path: Path,
    backbone_checkpoint_path: Path,
    expected_oof_summary_sha256: str | None = None,
    expected_fold_summary_sha256: str | None = None,
    expected_head_checkpoint_sha256: str | None = None,
    expected_backbone_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate one trained head/backbone fold without opening an image."""

    seed = int(pepd_seed)
    require(seed in SUPPORTED_PEPD_SEEDS, "unsupported enhanced-V5 OOF seed")
    final_path = guard_public_path(
        oof_summary_path, label="enhanced-V5 OOF final summary"
    )
    if expected_oof_summary_sha256 is not None:
        require(
            sha256_file(final_path) == expected_oof_summary_sha256,
            "enhanced-V5 OOF final summary hash drift",
        )
    final = strict_json(final_path)
    require(final.get("schema_version") == 1, "enhanced-V5 OOF schema drift")
    require(final.get("protocol") == OOF_PROTOCOL, "enhanced-V5 OOF protocol drift")
    require(final.get("status") == "complete", "enhanced-V5 OOF run is incomplete")
    require(final.get("mode") == "formal", "enhanced-V5 OOF run is not formal")
    scope = final.get("scope") or {}
    for key in (
        "field_samples_read",
        "public_test_samples_read",
        "confirmatory_samples_read",
        "sealed_samples_read",
    ):
        require(scope.get(key) == 0, f"enhanced-V5 OOF scope violation: {key}")
    strict = final.get("strict_oof") or {}
    require(strict.get("status") == "complete", "strict enhanced-V5 OOF is incomplete")
    overlap = strict.get("overlap_and_assignment_audit") or {}
    require(
        overlap.get("all_rows_jointly_unseen_by_pepd_and_head") is True,
        "enhanced-V5 OOF joint-unseen proof is absent",
    )
    require(
        int(overlap.get("eligible_union_samples", 0)) == 4380
        and int(overlap.get("eligible_union_groups", 0)) == 197,
        "enhanced-V5 OOF union inventory drift",
    )

    matches = [
        row
        for row in strict.get("folds") or []
        if int(row.get("pepd_seed", -1)) == seed
    ]
    require(len(matches) == 1, f"enhanced-V5 OOF fold {seed} is absent/duplicated")
    fold_entry = matches[0]
    fold_summary_path = guard_public_path(
        Path(str(fold_entry.get("summary") or "")),
        label=f"enhanced-V5 OOF fold {seed} summary",
    )
    require(
        sha256_file(fold_summary_path) == fold_entry.get("summary_sha256"),
        f"enhanced-V5 OOF fold {seed} summary hash drift",
    )
    if expected_fold_summary_sha256 is not None:
        require(
            sha256_file(fold_summary_path) == expected_fold_summary_sha256,
            f"enhanced-V5 OOF fold {seed} expected summary hash drift",
        )
    fold = strict_json(fold_summary_path)
    require(fold.get("protocol") == OOF_PROTOCOL, "enhanced-V5 fold protocol drift")
    require(fold.get("status") == "complete", "enhanced-V5 fold is incomplete")
    require(int(fold.get("pepd_seed", -1)) == seed, "enhanced-V5 fold seed drift")
    require(
        fold.get("jointly_unseen_contract") is True,
        "enhanced-V5 fold joint-unseen contract is absent",
    )
    split = fold.get("split_identity") or {}
    require(int(split.get("group_overlap", -1)) == 0, "enhanced-V5 fold group overlap")

    head_path = guard_public_path(
        head_checkpoint_path, label=f"enhanced-V5 OOF fold {seed} head"
    )
    artifacts = fold.get("artifacts") or {}
    require(
        Path(str(artifacts.get("checkpoint") or "")).resolve(strict=True) == head_path,
        "enhanced-V5 head path differs from fold summary",
    )
    require(
        sha256_file(head_path) == artifacts.get("checkpoint_sha256"),
        "enhanced-V5 head hash drift",
    )
    if expected_head_checkpoint_sha256 is not None:
        require(
            sha256_file(head_path) == expected_head_checkpoint_sha256,
            "enhanced-V5 expected head hash drift",
        )

    backbone_path = guard_public_path(
        backbone_checkpoint_path,
        label=f"enhanced-V5 OOF fold {seed} PEPD backbone",
    )
    backbone = fold.get("pepd_checkpoint") or {}
    require(
        Path(str(backbone.get("path") or "")).resolve(strict=True) == backbone_path,
        "enhanced-V5 backbone path differs from fold summary",
    )
    require(
        sha256_file(backbone_path) == backbone.get("sha256"),
        "enhanced-V5 backbone hash drift",
    )
    if expected_backbone_checkpoint_sha256 is not None:
        require(
            sha256_file(backbone_path) == expected_backbone_checkpoint_sha256,
            "enhanced-V5 expected backbone hash drift",
        )
    require(
        fold.get("split_identity", {}).get("checkpoint_sha256")
        == sha256_file(backbone_path),
        "enhanced-V5 split/backbone hash drift",
    )
    return {
        "protocol": PROVIDER_PROTOCOL,
        "pepd_seed": seed,
        "oof_summary": _binding(final_path, label="enhanced-V5 OOF final summary"),
        "fold_summary": _binding(
            fold_summary_path, label=f"enhanced-V5 OOF fold {seed} summary"
        ),
        "head_checkpoint": _binding(
            head_path, label=f"enhanced-V5 OOF fold {seed} head"
        ),
        "backbone_checkpoint": _binding(
            backbone_path, label=f"enhanced-V5 OOF fold {seed} PEPD backbone"
        ),
        "head_training_seed": int(fold["head_seed"]),
        "assigned_samples": int(split["assigned_samples"]),
        "assigned_groups": int(split["assigned_groups"]),
        "sample_assignment_sha256": str(overlap["sample_assignment_sha256"]),
        "group_assignment_sha256": str(overlap["group_assignment_sha256"]),
        "group_overlap": 0,
        "jointly_unseen_contract": True,
    }


class FrozenEnhancedV5OOFGeometryProvider:
    """CPU inference for one authenticated PEPD + enhanced-V5 OOF fold."""

    def __init__(
        self,
        *,
        oof_summary_path: Path,
        pepd_seed: int,
        head_checkpoint_path: Path,
        backbone_checkpoint_path: Path,
        expected_oof_summary_sha256: str | None = None,
        expected_fold_summary_sha256: str | None = None,
        expected_head_checkpoint_sha256: str | None = None,
        expected_backbone_checkpoint_sha256: str | None = None,
    ) -> None:
        self.device = torch.device("cpu")
        binding = authenticate_oof_fold(
            oof_summary_path=oof_summary_path,
            pepd_seed=pepd_seed,
            head_checkpoint_path=head_checkpoint_path,
            backbone_checkpoint_path=backbone_checkpoint_path,
            expected_oof_summary_sha256=expected_oof_summary_sha256,
            expected_fold_summary_sha256=expected_fold_summary_sha256,
            expected_head_checkpoint_sha256=expected_head_checkpoint_sha256,
            expected_backbone_checkpoint_sha256=expected_backbone_checkpoint_sha256,
        )
        head_checkpoint = torch.load(
            Path(binding["head_checkpoint"]["path"]),
            map_location="cpu",
            weights_only=False,
        )
        require(isinstance(head_checkpoint, Mapping), "enhanced-V5 head is malformed")
        require(head_checkpoint.get("protocol") == OOF_PROTOCOL, "head protocol drift")
        require(head_checkpoint.get("status") == "complete", "head is incomplete")
        require(
            int(head_checkpoint.get("pepd_seed", -1)) == int(pepd_seed),
            "head PEPD seed drift",
        )
        require(
            head_checkpoint.get("pepd_checkpoint_sha256")
            == binding["backbone_checkpoint"]["sha256"],
            "head/backbone checkpoint drift",
        )
        require(isinstance(head_checkpoint.get("head_state"), Mapping), "head state absent")

        backbone_path = Path(binding["backbone_checkpoint"]["path"])
        self.backbone, backbone_identity = public_v5.load_pepd(
            backbone_path, self.device
        )
        require(
            backbone_identity.get("sha256")
            == binding["backbone_checkpoint"]["sha256"],
            "loaded PEPD backbone hash drift",
        )
        self.head = build_head().to(self.device)
        self.head.load_state_dict(head_checkpoint["head_state"], strict=True)
        self.backbone.eval()
        self.head.eval()
        self.identity = {
            "protocol": PROVIDER_PROTOCOL,
            "provider": "fold_matched_pepd_plus_enhanced_v5_oof_head",
            "execution_device": "cpu",
            **binding,
            "backbone_loader_identity": backbone_identity,
        }

    @torch.inference_mode()
    def predict(self, canonical_roi_bgr: np.ndarray) -> GeometryProviderResult:
        interpolation = (
            cv2.INTER_AREA
            if max(canonical_roi_bgr.shape[:2]) > 256
            else cv2.INTER_LINEAR
        )
        model_image = cv2.resize(
            canonical_roi_bgr, (256, 256), interpolation=interpolation
        )
        tensor = normalized_rgb_tensor(model_image)[None].to(self.device)
        features = self.backbone.forward_multiscale_features(tensor)
        pooled = self.backbone.direction_features(features.c5)
        pivot_logits = self.backbone.pivot_head(features.c5)
        direction = decode_probabilistic_pivot_direction(
            pivot_logits,
            self.backbone.vector_head(pooled),
            self.backbone.angle_head(pooled),
            self.backbone.log_variance_head(pooled),
        )
        endpoint = self.head(features.c2, features.c5)

        heat_height, heat_width = pivot_logits.shape[-2:]
        pivot = direction.pivot_xy[0].detach().cpu().numpy().astype(np.float64)
        pivot_normalized = (
            float(pivot[0] / max(heat_width - 1, 1)),
            float(pivot[1] / max(heat_height - 1, 1)),
        )
        start = tuple(float(value) for value in endpoint.start_xy[0].cpu().tolist())
        end = tuple(float(value) for value in endpoint.end_xy[0].cpu().tolist())
        confidence = float(endpoint.gate_confidence[0].cpu())
        hint = GeometryHint(
            pivot_xy=pivot_normalized,
            start_xy=start,
            end_xy=end,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            source="fold_matched_pepd_plus_enhanced_v5_oof_head",
        ).validate()
        return GeometryProviderResult(
            hint=hint,
            telemetry={
                "protocol": PROVIDER_PROTOCOL,
                "pepd_seed": int(self.identity["pepd_seed"]),
                "pivot_peak": float(direction.pivot_peak[0].cpu()),
                "pivot_valid": bool(direction.valid[0].cpu()),
                "endpoint_peak": endpoint.endpoint_peak[0].cpu().tolist(),
                "endpoint_entropy": endpoint.endpoint_entropy[0].cpu().tolist(),
                "endpoint_separation": float(endpoint.endpoint_separation[0].cpu()),
                "gate_confidence": confidence,
                "uncertainty_score": float(endpoint.uncertainty_score[0].cpu()),
                "telemetry_valid": bool(endpoint.telemetry_valid[0].cpu()),
                "endpoint_coordinate_disagreement": float(
                    endpoint.endpoint_coordinate_disagreement[0].cpu()
                ),
                "model_input_shape": list(model_image.shape),
            },
        )


__all__ = [
    "OOF_PROTOCOL",
    "PROVIDER_PROTOCOL",
    "SUPPORTED_PEPD_SEEDS",
    "FrozenEnhancedV5OOFGeometryProvider",
    "authenticate_oof_fold",
]
