"""Evaluate one matched-split VDN seed on GeoPRR's fixed SyncG pixels.

VDN predicts pointer direction from pixels only.  Ordered SyncG scale endpoints
and the annotated pivot are used offline to convert that direction to normalized
reading progress.  The result is therefore a deliberately stringent,
annotation-assisted component comparison, not a deployable end-to-end VDN
system.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from experiments import run_vdn_oracle_reference_component as oracle
from experiments.vdn_baseline import (
    build_vdn_model,
    image_angle_from_direction,
    normalized_bgr_tensor,
    predict_directions,
    verify_vdn_source,
)


PROTOCOL = "geoprr_vdn_matched_annotation_component_v1"
EVALUATION_ROLE = "matched_open_source_annotation_assisted_component"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class MatchedVDNDirectionPredictor:
    """Thin inference adapter for one terminal matched-split VDN checkpoint."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        vdn_source: Path,
        seed: int,
        expected_epochs: int,
        device: str,
        amp_enabled: bool,
    ) -> None:
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self._amp_enabled = bool(amp_enabled and self._device.type == "cuda")
        checkpoint = Path(checkpoint).resolve()
        _require(checkpoint.is_file(), f"VDN checkpoint is missing: {checkpoint}")
        verify_vdn_source(Path(vdn_source))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _require(isinstance(payload, Mapping), "VDN checkpoint is not a mapping")
        signature = payload.get("signature")
        _require(isinstance(signature, Mapping), "VDN checkpoint signature is missing")
        _require(
            signature.get("matched_geoprr_split") is True,
            "VDN checkpoint was not trained on the matched GeoPRR split",
        )
        _require(int(signature.get("seed", -1)) == int(seed), "VDN seed differs")
        _require(
            int(signature.get("epochs", -1)) == int(expected_epochs)
            and int(payload.get("epoch", -1)) == int(expected_epochs),
            "VDN terminal epoch differs from the requested fixed budget",
        )
        _require(
            int(signature.get("train_samples", -1)) == 12_866
            and int(signature.get("validation_samples", -1)) == 1_576,
            "VDN matched train/validation roster differs",
        )
        self.native_input_size = int(signature.get("image_size", 0))
        _require(
            self.native_input_size >= 32 and self.native_input_size % 32 == 0,
            "VDN native input size is invalid",
        )
        state = payload.get("model_state")
        _require(isinstance(state, Mapping) and bool(state), "VDN model state is absent")
        model = build_vdn_model(
            Path(vdn_source),
            image_size=self.native_input_size,
            imagenet_pretrained=False,
        )
        model.load_state_dict(state, strict=True)
        model.to(self._device)
        model.eval()
        self._model = model

    @torch.inference_mode()
    def predict_batch(
        self, requests: Sequence[oracle.DirectionRequest]
    ) -> Sequence[Mapping[str, Any]]:
        if not requests:
            return []
        tensors: list[torch.Tensor] = []
        for request in requests:
            image = np.asarray(request.image_bgr)
            _require(
                image.ndim == 3 and image.shape[2] == 3,
                f"{request.source.sample_id}: invalid conditioned ROI",
            )
            resized = cv2.resize(
                image,
                (self.native_input_size, self.native_input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            tensors.append(normalized_bgr_tensor(resized))
        inputs = torch.stack(tensors).to(self._device, non_blocking=True)
        with torch.amp.autocast(
            self._device.type,
            enabled=self._amp_enabled,
        ):
            heatmaps, vector_maps = self._model(inputs)
        directions, peaks, valid = predict_directions(
            heatmaps.float(), vector_maps.float()
        )
        directions_cpu = directions.detach().cpu().numpy()
        peaks_cpu = peaks.detach().cpu().numpy()
        valid_cpu = valid.detach().cpu().numpy()
        reference_hash = oracle._reference_sha256(oracle._DIRECTION_ONLY_REFERENCE)
        records: list[dict[str, Any]] = []
        for index in range(len(requests)):
            peak = float(peaks_cpu[index])
            if not bool(valid_cpu[index]) or not math.isfinite(peak):
                records.append(
                    {
                        "status": False,
                        "prediction_progress": None,
                        "failure_code": "invalid_vdn_direction",
                        "reference_input_sha256": reference_hash,
                        "telemetry": {"heatmap_peak": peak},
                    }
                )
                continue
            pointer_angle = image_angle_from_direction(directions_cpu[index])
            records.append(
                {
                    "status": True,
                    "prediction_progress": None,
                    "failure_code": None,
                    "reference_input_sha256": reference_hash,
                    "telemetry": {
                        "pointer_angle": float(pointer_angle),
                        "heatmap_peak": peak,
                    },
                }
            )
        return records


def evaluate(
    *,
    checkpoint: Path,
    vdn_source: Path,
    roi_manifest: Path,
    syncg_manifest: Path,
    output: Path,
    seed: int,
    expected_epochs: int,
    device: str,
    batch_size: int,
    amp_enabled: bool,
) -> int:
    output = Path(output).resolve()
    _require(not output.exists(), f"VDN evaluation output already exists: {output}")
    method = f"vdn_geoprr_matched_terminal_seed_{int(seed)}"
    oracle.METHOD = method
    oracle.PROTOCOL = PROTOCOL
    oracle.EVALUATION_ROLE = EVALUATION_ROLE

    def predictor_factory(*, device: str):
        return MatchedVDNDirectionPredictor(
            checkpoint=checkpoint,
            vdn_source=vdn_source,
            seed=seed,
            expected_epochs=expected_epochs,
            device=device,
            amp_enabled=amp_enabled,
        )

    return oracle.run_component(
        roi_manifest_path=roi_manifest,
        syncg_manifest_path=syncg_manifest,
        output_path=output,
        device=device,
        batch_size=batch_size,
        expected_samples=1_558,
        expected_groups=70,
        expected_ids_sha256=None,
        expected_roi_manifest_sha256=None,
        expected_source_manifest_sha256=None,
        predictor_factory=predictor_factory,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vdn-source", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, required=True)
    parser.add_argument("--syncg-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    rows = evaluate(
        checkpoint=args.checkpoint,
        vdn_source=args.vdn_source,
        roi_manifest=args.roi_manifest,
        syncg_manifest=args.syncg_manifest,
        output=args.output,
        seed=args.seed,
        expected_epochs=args.expected_epochs,
        device=args.device,
        batch_size=args.batch_size,
        amp_enabled=not args.no_amp,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "protocol": PROTOCOL,
                "seed": int(args.seed),
                "rows": rows,
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EVALUATION_ROLE", "PROTOCOL", "MatchedVDNDirectionPredictor", "evaluate"]
