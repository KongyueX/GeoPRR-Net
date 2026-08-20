"""Modern lightweight direct-regression baselines for the paper.

The formal training command consumes only the fixed SyncG scene-fit split and
uses the same ROI, 256-pixel input, augmentation, SmoothL1 loss, AdamW cosine
schedule, terminal-epoch policy, and seeds as the existing direct ResNet-18
runner.  The only intended experimental difference is the torchvision
backbone: MobileNetV3-Large or EfficientNet-B0.

Field photographs are not accepted by the training interface.  They can be
evaluated later through ``load_checkpoint_predictor`` or the shared prediction
and scoring interfaces after a terminal checkpoint has been produced.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)

from experiments import robustness_degradations
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MANIFEST,
    DEFAULT_SEEDS,
    DirectProgressDataset,
    DirectProgressError,
    _configure_reproducibility,
    _epoch,
    _loader,
    load_training_samples,
)
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest as load_plain_manifest,
)
from experiments.run_cagh_v5_solver_gated_screen import augmentation_config
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


PROTOCOL: Final[str] = "syncg_lightweight_regression_baselines_v1"
IMAGE_SIZE: Final[int] = 256
DEFAULT_SCENE_SPLIT: Final[Path] = Path(
    "C:/pointer_read/syncg_scene_disjoint_clean_v1/split.json"
)
EXPECTED_FIT_SAMPLES: Final[int] = 14_442
EXPECTED_FIT_SCENES: Final[int] = 131
EXPECTED_HOLDOUT_SAMPLES: Final[int] = 1_558
ARCHITECTURES: Final[tuple[str, ...]] = (
    "mobilenet_v3_large",
    "efficientnet_b0",
)


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    paper_name: str
    weights_name: str
    weights: Any
    builder: Callable[..., nn.Module]


BACKBONE_SPECS: Final[dict[str, BackboneSpec]] = {
    "mobilenet_v3_large": BackboneSpec(
        paper_name="MobileNetV3-Large",
        weights_name=(
            "torchvision.MobileNet_V3_Large_Weights.IMAGENET1K_V2"
        ),
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        builder=mobilenet_v3_large,
    ),
    "efficientnet_b0": BackboneSpec(
        paper_name="EfficientNet-B0",
        weights_name="torchvision.EfficientNet_B0_Weights.IMAGENET1K_V1",
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
        builder=efficientnet_b0,
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectProgressError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def method_id(architecture: str, seed: int) -> str:
    _require(architecture in BACKBONE_SPECS, f"unsupported architecture: {architecture}")
    return f"{architecture}_seed_{int(seed)}"


class LightweightProgressRegressor(nn.Module):
    """Official ImageNet backbone followed by one sigmoid progress output."""

    def __init__(
        self,
        architecture: str,
        *,
        imagenet_pretrained: bool = True,
    ) -> None:
        super().__init__()
        _require(architecture in BACKBONE_SPECS, f"unsupported architecture: {architecture}")
        self.architecture = architecture
        self.imagenet_pretrained = bool(imagenet_pretrained)
        spec = BACKBONE_SPECS[architecture]
        self.backbone = spec.builder(
            weights=spec.weights if self.imagenet_pretrained else None
        )
        classifier = getattr(self.backbone, "classifier", None)
        _require(
            isinstance(classifier, nn.Sequential) and len(classifier) >= 1,
            f"{architecture}: torchvision classifier layout is unsupported",
        )
        final = classifier[-1]
        _require(
            isinstance(final, nn.Linear),
            f"{architecture}: final classifier is not linear",
        )
        classifier[-1] = nn.Linear(int(final.in_features), 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(image).squeeze(1))


def parameter_count(architecture: str) -> int:
    model = LightweightProgressRegressor(architecture, imagenet_pretrained=False)
    return sum(parameter.numel() for parameter in model.parameters())


def _validate_syncg_scene_fit(fit_samples: Sequence[Any], roster: Any) -> None:
    """Reject an accidental non-SyncG or non-scene-disjoint training roster."""

    _require(roster.scene_disjoint, "formal baseline split must be scene-disjoint")
    _require(
        len(fit_samples) == EXPECTED_FIT_SAMPLES,
        f"formal SyncG fit count must be {EXPECTED_FIT_SAMPLES}",
    )
    _require(
        len({sample.scene_stem for sample in fit_samples}) == EXPECTED_FIT_SCENES,
        f"formal SyncG fit scene count must be {EXPECTED_FIT_SCENES}",
    )
    _require(
        len(roster.validation_ids) == EXPECTED_HOLDOUT_SAMPLES,
        f"formal SyncG holdout count must be {EXPECTED_HOLDOUT_SAMPLES}",
    )


def train(
    *,
    architecture: str,
    manifest_path: Path,
    split_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Train a fixed terminal iterate without opening holdout pixels or labels."""

    _require(architecture in BACKBONE_SPECS, f"unsupported architecture: {architecture}")
    _require(epochs >= 1 and batch_size >= 1 and workers >= 0, "invalid training sizes")
    _require(learning_rate > 0.0 and weight_decay >= 0.0, "invalid optimizer values")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"refusing to overwrite checkpoint: {output}")

    fit_samples, roster = load_training_samples(manifest_path, split_path)
    _validate_syncg_scene_fit(fit_samples, roster)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    fit_dataset = DirectProgressDataset(fit_samples, training=True, seed=seed)
    model = LightweightProgressRegressor(
        architecture,
        imagenet_pretrained=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch_index in range(epochs):
        epoch_started = time.perf_counter()
        fit_dataset.set_epoch(epoch_index)
        loader = _loader(
            fit_dataset,
            batch_size=batch_size,
            shuffle=True,
            workers=workers,
            seed=seed + epoch_index,
            cuda=device.type == "cuda",
        )
        metrics = _epoch(
            model,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        history.append(
            {
                "epoch": epoch_index + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - epoch_started,
                "train": metrics,
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        scheduler.step()

    spec = BACKBONE_SPECS[architecture]
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": architecture,
        "paper_name": spec.paper_name,
        "pretrained_weights": spec.weights_name,
        "image_size": IMAGE_SIZE,
        "seed": int(seed),
        "split_protocol": roster.protocol,
        "scene_disjoint": roster.scene_disjoint,
        "train_samples": len(fit_samples),
        "holdout_samples": len(roster.validation_ids),
        "holdout_access_during_training": (
            "IDs/count only; no target, bbox, image path, or image"
        ),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "checkpoint_selection": "terminal_fixed_epoch",
        "loss": "smooth_l1_beta_0.05",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
        },
        "augmentation": {
            "name": "matched_cagh_v5_photo_and_geometry",
            "configuration": augmentation_config(),
        },
        "training_elapsed_seconds": time.perf_counter() - training_started,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "method": method_id(architecture, seed),
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "terminal_train_nmae": history[-1]["train"]["nmae"],
        "training_elapsed_seconds": checkpoint["training_elapsed_seconds"],
        "train_samples": len(fit_samples),
        "holdout_samples": len(roster.validation_ids),
        "scene_disjoint": roster.scene_disjoint,
    }


def load_checkpoint_predictor(
    checkpoint_path: Path,
    *,
    device_name: str,
) -> tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "checkpoint is not an object")
    _require(checkpoint.get("protocol") == PROTOCOL, "checkpoint protocol mismatch")
    architecture = str(checkpoint.get("architecture") or "")
    _require(architecture in BACKBONE_SPECS, "checkpoint architecture is unsupported")
    _require(
        checkpoint.get("pretrained_weights")
        == BACKBONE_SPECS[architecture].weights_name,
        "checkpoint ImageNet initialization metadata mismatch",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "checkpoint model state is missing")
    seed = int(checkpoint["seed"])
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = LightweightProgressRegressor(
        architecture,
        imagenet_pretrained=False,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    def predict(images_bgr: Sequence[np.ndarray]) -> list[float]:
        _require(bool(images_bgr), "prediction batch is empty")
        tensors = [
            normalized_rgb_tensor(direct_resize_whole_roi(image, size=IMAGE_SIZE))
            for image in images_bgr
        ]
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            values = model(batch).detach().cpu().tolist()
        results = [float(value) for value in values]
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in results),
            "model returned invalid progress",
        )
        return results

    return method_id(architecture, seed), predict


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
) -> int:
    """Write label-free predictions in the established paper JSONL schema."""

    selected_conditions = tuple(conditions)
    _require(bool(selected_conditions), "no evaluation conditions selected")
    _require(
        len(selected_conditions) == len(set(selected_conditions))
        and set(selected_conditions) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_plain_manifest(manifest_path)
    method, predictor = load_checkpoint_predictor(
        checkpoint_path,
        device_name=device_name,
    )
    output = Path(output_path).resolve()
    _require(output != Path(manifest_path).resolve(), "output cannot overwrite manifest")
    _require(not output.exists(), f"refusing to overwrite predictions: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            conditioned: list[np.ndarray] = []
            condition_hashes: list[str] = []
            for condition in selected_conditions:
                image, _metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                image = np.ascontiguousarray(image)
                conditioned.append(image)
                condition_hashes.append(canonical_roi_pixel_sha256(image))
            try:
                predicted = predictor(conditioned)
                _require(
                    len(predicted) == len(conditioned),
                    "prediction batch length mismatch",
                )
                progress_values: list[float | None] = [float(value) for value in predicted]
                _require(
                    all(
                        math.isfinite(value) and 0.0 <= value <= 1.0
                        for value in progress_values
                    ),
                    "prediction outside [0,1]",
                )
                failure_codes: list[str | None] = [None] * len(conditioned)
            except Exception as exc:
                progress_values = [None] * len(conditioned)
                failure_codes = [
                    f"model_exception:{type(exc).__name__}"
                ] * len(conditioned)
            for condition, condition_hash, progress, failure in zip(
                selected_conditions,
                condition_hashes,
                progress_values,
                failure_codes,
                strict=True,
            ):
                passed = progress is not None and failure is None
                row = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": source.sample_id,
                    "method": method,
                    "condition": condition,
                    "robustness_seed": ROBUSTNESS_SEED,
                    "status": "pass" if passed else "fail",
                    "normalized_progress": progress if passed else None,
                    "failure_code": None if passed else failure,
                    "roi_png_sha256": source.roi_png_sha256,
                    "roi_pixel_sha256": source.roi_pixel_sha256,
                    "condition_pixel_sha256": condition_hash,
                }
                _require(set(row) == set(OUTPUT_KEYS), "prediction output schema drift")
                stream.write(_canonical_json_bytes(row).decode("utf-8") + "\n")
                count += 1
    return count


def score_predictions(
    *,
    architecture: str,
    prediction_paths: Sequence[Path],
    manifest_path: Path,
    validation_ids_path: Path,
    output_path: Path,
    seeds: Sequence[int],
    conditions: Sequence[str],
    bootstrap_replicates: int = 2_000,
) -> dict[str, Any]:
    """Use the established failure-penalized, group-bootstrap scorer."""

    from experiments import score_cagh_v5_plain_paper_batch as scorer

    methods = tuple(method_id(architecture, seed) for seed in seeds)
    _require(len(methods) == len(set(methods)) and bool(methods), "seed roster is invalid")
    _require(len(methods) in (1, 3), "score expects one or three seeds")
    validation_ids = scorer.load_validation_ids(validation_ids_path)
    targets = scorer.load_targets(manifest_path, validation_ids)
    value = scorer.score(
        predictions_path=tuple(prediction_paths),
        manifest_path=manifest_path,
        validation_ids_path=validation_ids_path,
        methods=methods,
        conditions=tuple(conditions),
        full_seed_methods=methods if len(methods) == 3 else (),
        expected_samples=len(validation_ids),
        expected_groups=len({target.group_id for target in targets}),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=ROBUSTNESS_SEED,
    )
    _write_json(output_path, value)
    return value


def timing_smoke(
    *,
    architectures: Sequence[str] = ARCHITECTURES,
    device_name: str = "cpu",
    image_size: int = IMAGE_SIZE,
    batch_size: int = 1,
    iterations: int = 1,
) -> dict[str, Any]:
    """Run unpretrained forward/backward steps; this is not a training benchmark."""

    _require(image_size >= 32 and batch_size >= 1 and iterations >= 1, "invalid smoke sizes")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    results: list[dict[str, Any]] = []
    for architecture in architectures:
        model = LightweightProgressRegressor(
            architecture,
            imagenet_pretrained=False,
        ).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        images = torch.rand(batch_size, 3, image_size, image_size, device=device)
        targets = torch.rand(batch_size, device=device)
        elapsed: list[float] = []
        final_loss = 0.0
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            predictions = model(images)
            loss = F.smooth_l1_loss(predictions, targets, beta=0.05)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
            final_loss = float(loss.detach().cpu())
        results.append(
            {
                "architecture": architecture,
                "paper_name": BACKBONE_SPECS[architecture].paper_name,
                "official_weights_for_formal_training": (
                    BACKBONE_SPECS[architecture].weights_name
                ),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "batch_size": batch_size,
                "image_size": image_size,
                "iterations": iterations,
                "mean_step_seconds": sum(elapsed) / len(elapsed),
                "final_loss": final_loss,
            }
        )
        del model, optimizer, images, targets
    return {
        "status": "complete",
        "device": str(device),
        "note": "Unpretrained single-step smoke; not a formal throughput estimate.",
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser("train")
    training.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    training.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    training.add_argument("--split", type=Path, default=DEFAULT_SCENE_SPLIT)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, choices=DEFAULT_SEEDS, required=True)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    training.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)

    prediction = subparsers.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cuda:0")
    prediction.add_argument("--conditions", choices=("all", "clean"), default="all")

    scoring = subparsers.add_parser("score")
    scoring.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    scoring.add_argument("--predictions", type=Path, nargs="+", required=True)
    scoring.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    scoring.add_argument("--validation-ids", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--seeds", type=int, nargs="+", required=True)
    scoring.add_argument("--conditions", choices=("all", "clean"), default="all")
    scoring.add_argument("--bootstrap-replicates", type=int, default=2_000)

    smoke = subparsers.add_parser("timing-smoke")
    smoke.add_argument("--architectures", choices=ARCHITECTURES, nargs="+", default=ARCHITECTURES)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    smoke.add_argument("--batch-size", type=int, default=1)
    smoke.add_argument("--iterations", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = train(
            architecture=args.architecture,
            manifest_path=args.manifest,
            split_path=args.split,
            output_path=args.output,
            seed=args.seed,
            device_name=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.command == "predict":
        conditions = CONDITIONS if args.conditions == "all" else ("clean",)
        rows = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            device_name=args.device,
            conditions=conditions,
        )
        result = {
            "status": "complete",
            "output": str(Path(args.output).resolve()),
            "rows": rows,
            "conditions": list(conditions),
        }
    elif args.command == "score":
        conditions = CONDITIONS if args.conditions == "all" else ("clean",)
        value = score_predictions(
            architecture=args.architecture,
            prediction_paths=args.predictions,
            manifest_path=args.manifest,
            validation_ids_path=args.validation_ids,
            output_path=args.output,
            seeds=args.seeds,
            conditions=conditions,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        result = {
            "status": "complete",
            "output": str(Path(args.output).resolve()),
            "samples": value["identity"]["samples"],
            "methods": value["identity"]["methods"],
            "conditions": value["identity"]["conditions"],
        }
    else:
        result = timing_smoke(
            architectures=args.architectures,
            device_name=args.device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            iterations=args.iterations,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHITECTURES",
    "BACKBONE_SPECS",
    "IMAGE_SIZE",
    "LightweightProgressRegressor",
    "load_checkpoint_predictor",
    "main",
    "method_id",
    "parameter_count",
    "run_prediction",
    "score_predictions",
    "timing_smoke",
    "train",
]
