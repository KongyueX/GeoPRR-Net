"""Train the public group-disjoint gauge numeric detector and CTC recognizer.

Formal mode is intentionally one seed at a time.  It selects checkpoints only
on the disjoint calibration groups and evaluates the untouched validation
groups after selection.  Smoke mode performs real CPU forward/backward steps
on tiny public subsets and never downloads pretrained weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.syncg_numeric_ocr import (
    BLANK_INDEX,
    CHECKPOINT_PROTOCOL,
    PROTOCOL,
    VOCABULARY,
    GaugeTextDetector,
    OCRCorpus,
    SyncGTextDetectionDataset,
    SyncGTextRecognitionImageDataset,
    TinyCTCRecognizer,
    canonical_sha256,
    edit_distance,
    greedy_ctc_decode,
    recognition_image_collate,
    sha256_file,
    text_detection_loss,
)
from experiments.ocr_training_resume import (
    canonical_sha256 as resume_signature_sha256,
    load_epoch_journal,
    save_epoch_journal,
)


SAFE_OUTPUT_ROOT = Path(r"C:\pointer_read").resolve()
DEFAULT_CORPUS = SAFE_OUTPUT_ROOT / "syncg_numeric_ocr_public_v1"
DEFAULT_OUTPUT = SAFE_OUTPUT_ROOT / "syncg_numeric_ocr_runs"
TRAINER_PATH = Path(__file__).resolve()
IMPLEMENTATION_PATH = PROJECT_ROOT / "experiments/syncg_numeric_ocr.py"
RESUME_UTILITY_PATH = PROJECT_ROOT / "experiments/ocr_training_resume.py"
DETECTOR_RESUME_PROTOCOL = "syncg_public_numeric_ocr_detector_resume_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--component", choices=("both", "detector", "recognizer"), default="both")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--detector-image-size", type=int, default=512)
    parser.add_argument("--detector-batch-size", type=int, default=12)
    parser.add_argument("--detector-epochs", type=int, default=12)
    parser.add_argument("--detector-learning-rate", type=float, default=2e-4)
    parser.add_argument("--detector-frozen-epochs", type=int, default=2)
    parser.add_argument("--detector-validation-limit", type=int, default=800)
    parser.add_argument("--detector-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recognizer-image-batch-size", type=int, default=16)
    parser.add_argument("--recognizer-epochs", type=int, default=18)
    parser.add_argument("--recognizer-learning-rate", type=float, default=6e-4)
    parser.add_argument("--decimal-probability", type=float, default=0.20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only an incomplete formal run from authenticated artifacts",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    collate_fn: Any = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        persistent_workers=False,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def detection_collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in rows]),
        "target": torch.stack([row["target"] for row in rows]),
        "sample_ids": [str(row["sample_id"]) for row in rows],
    }


@torch.inference_mode()
def evaluate_recognizer(
    model: TinyCTCRecognizer,
    dataset: SyncGTextRecognitionImageDataset,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    model.eval()
    loader = _loader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=False,
        seed=0,
        workers=workers,
        collate_fn=recognition_image_collate,
    )
    total = exact = characters = errors = parseable = 0
    confidence: list[float] = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        predictions, scores = greedy_ctc_decode(logits.cpu())
        for prediction, truth, score in zip(predictions, batch["texts"], scores, strict=True):
            total += 1
            exact += prediction == truth
            characters += len(truth)
            errors += edit_distance(prediction, truth)
            try:
                value = float(prediction)
                parseable += math.isfinite(value)
            except ValueError:
                pass
            confidence.append(score)
    _require(total == dataset.token_count, "recognizer evaluation inventory drift")
    return {
        "tokens": total,
        "source_images": len(dataset),
        "exact_accuracy": exact / total,
        "character_accuracy": 1.0 - errors / max(characters, 1),
        "parseable_fraction": parseable / total,
        "mean_confidence": float(np.mean(confidence)),
    }


@torch.inference_mode()
def evaluate_detector(
    model: GaugeTextDetector,
    dataset: SyncGTextDetectionDataset,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    model.eval()
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        workers=workers,
        collate_fn=detection_collate,
    )
    intersections = predicted = targets = samples = 0.0
    loss_sum = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        logits = model(image)
        loss = text_detection_loss(logits, target)
        binary = torch.sigmoid(logits) >= 0.40
        intersections += float((binary & (target >= 0.5)).sum())
        predicted += float(binary.sum())
        targets += float((target >= 0.5).sum())
        loss_sum += float(loss) * image.shape[0]
        samples += image.shape[0]
    _require(int(samples) == len(dataset), "detector evaluation inventory drift")
    return {
        "samples": int(samples),
        "loss": loss_sum / samples,
        "pixel_dice_at_0_40": (2.0 * intersections + 1.0) / (predicted + targets + 1.0),
        "predicted_positive_fraction": predicted / (samples * dataset.image_size**2),
        "target_positive_fraction": targets / (samples * dataset.image_size**2),
    }


def train_recognizer(
    corpus: OCRCorpus,
    *,
    args: argparse.Namespace,
    device: torch.device,
    smoke: bool,
) -> tuple[TinyCTCRecognizer, Mapping[str, Any]]:
    train_limit = 2 if smoke else None
    calibration_limit = 2 if smoke else None
    train = SyncGTextRecognitionImageDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="train",
        seed=args.seed,
        training=True,
        decimal_probability=args.decimal_probability,
        limit=train_limit,
    )
    calibration = SyncGTextRecognitionImageDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="calibration",
        seed=args.seed,
        training=False,
        limit=calibration_limit,
    )
    validation = SyncGTextRecognitionImageDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="validation",
        seed=args.seed,
        training=False,
        limit=calibration_limit,
    )
    model = TinyCTCRecognizer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.recognizer_learning_rate, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    epochs = 1 if smoke else args.recognizer_epochs
    image_batch_size = min(args.recognizer_image_batch_size, len(train))
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train.set_epoch(epoch)
        loader = _loader(
            train,
            batch_size=image_batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            workers=0 if smoke else args.workers,
            collate_fn=recognition_image_collate,
        )
        model.train()
        loss_sum = samples = synthesized = 0
        for batch in loader:
            image = batch["image"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(image)
            input_lengths = torch.full(
                (image.shape[0],), logits.shape[0], dtype=torch.long, device=device
            )
            loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths)
            _require(bool(torch.isfinite(loss)), "recognizer loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * image.shape[0]
            samples += image.shape[0]
            synthesized += int(batch["decimal_synthesized"].sum())
        calibration_metrics = evaluate_recognizer(
            model,
            calibration,
            device=device,
            batch_size=min(image_batch_size, len(calibration)),
            workers=0 if smoke else args.workers,
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / samples,
            "train_samples": samples,
            "decimal_synthesized_fraction": synthesized / samples,
            "calibration": calibration_metrics,
        }
        history.append(row)
        metric = float(calibration_metrics["exact_accuracy"])
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"numeric-ocr recognizer seed={args.seed} epoch={epoch}/{epochs} "
            f"loss={row['train_loss']:.5f} cal_exact={metric:.4f}",
            flush=True,
        )
    _require(best_state is not None, "recognizer produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    validation_metrics = evaluate_recognizer(
        model,
        validation,
        device=device,
        batch_size=min(image_batch_size, len(validation)),
        workers=0 if smoke else args.workers,
    )
    return model, {
        "history": history,
        "selection": "maximum calibration exact string accuracy; first tie",
        "best_calibration_exact_accuracy": best_metric,
        "validation": validation_metrics,
        "train_source_images": len(train),
        "calibration_source_images": len(calibration),
        "validation_source_images": len(validation),
        "train_tokens": train.token_count,
        "calibration_tokens": calibration.token_count,
        "validation_tokens": validation.token_count,
    }


def _corpus_identity(corpus: OCRCorpus) -> dict[str, Any]:
    return {
        "root": str(corpus.root),
        "summary_sha256": sha256_file(corpus.root / "summary.json"),
        "samples_sha256": corpus.summary["artifacts"]["samples_sha256"],
        "tokens_sha256": corpus.summary["artifacts"]["tokens_sha256"],
    }


def _detector_resume_signature(
    *, args: argparse.Namespace, corpus: OCRCorpus, image_size: int, pretrained_identity: str
) -> dict[str, Any]:
    return {
        "protocol": DETECTOR_RESUME_PROTOCOL,
        "component": "detector",
        "seed": int(args.seed),
        "corpus": _corpus_identity(corpus),
        "configuration": {
            "image_size": int(image_size),
            "batch_size": int(args.detector_batch_size),
            "epochs": int(args.detector_epochs),
            "learning_rate": float(args.detector_learning_rate),
            "frozen_epochs": int(args.detector_frozen_epochs),
            "validation_limit": int(args.detector_validation_limit),
            "pretrained": bool(args.detector_pretrained),
            "pretrained_identity": str(pretrained_identity),
            "architecture": "MobileNetV3SmallFPN",
            "annular_geometry_channels": 0,
        },
        "code": {
            "trainer_sha256": sha256_file(TRAINER_PATH),
            "implementation_sha256": sha256_file(IMPLEMENTATION_PATH),
            "resume_utility_sha256": sha256_file(RESUME_UTILITY_PATH),
        },
    }


def _load_completed_component(
    path: Path,
    *,
    component: str,
    args: argparse.Namespace,
    corpus: OCRCorpus,
) -> Mapping[str, Any]:
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    _require(isinstance(value, Mapping), f"completed {component} checkpoint is not a mapping")
    _require(value.get("protocol") == CHECKPOINT_PROTOCOL, f"completed {component} protocol drift")
    _require(value.get("status") == "complete", f"completed {component} is not formal")
    _require(value.get("component") == component, f"completed {component} component drift")
    _require(value.get("mode") == "formal", f"completed {component} mode drift")
    _require(int(value.get("seed", -1)) == int(args.seed), f"completed {component} seed drift")
    _require(value.get("vocabulary") == list(VOCABULARY), f"completed {component} vocabulary drift")
    _require(value.get("corpus") == _corpus_identity(corpus), f"completed {component} corpus drift")
    _require(isinstance(value.get("metrics"), Mapping), f"completed {component} metrics missing")
    _require(isinstance(value.get("state_dict"), Mapping), f"completed {component} state missing")
    return value


def train_detector(
    corpus: OCRCorpus,
    *,
    args: argparse.Namespace,
    device: torch.device,
    smoke: bool,
    resume_path: Path | None = None,
) -> tuple[GaugeTextDetector, Mapping[str, Any]]:
    image_size = 128 if smoke else args.detector_image_size
    limit = 2 if smoke else None
    validation_limit = 2 if smoke else args.detector_validation_limit
    train = SyncGTextDetectionDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="train",
        image_size=image_size,
        training=True,
        seed=args.seed,
        limit=limit,
    )
    calibration = SyncGTextDetectionDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="calibration",
        image_size=image_size,
        training=False,
        limit=validation_limit,
    )
    validation = SyncGTextDetectionDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="validation",
        image_size=image_size,
        training=False,
        limit=validation_limit,
    )
    pretrained = bool(args.detector_pretrained and not smoke)
    model = GaugeTextDetector(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.detector_learning_rate, weight_decay=1e-4)
    epochs = 1 if smoke else args.detector_epochs
    batch_size = min(args.detector_batch_size, len(train))
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -math.inf
    history: list[dict[str, Any]] = []
    start_epoch = 1
    elapsed_before = 0.0
    attempt_started = time.perf_counter()
    resume_signature = _detector_resume_signature(
        args=args,
        corpus=corpus,
        image_size=image_size,
        pretrained_identity=model.pretrained_identity,
    )
    if resume_path is not None and resume_path.is_file():
        _require(not smoke, "smoke detector cannot resume")
        restored = load_epoch_journal(
            resume_path,
            expected_protocol=DETECTOR_RESUME_PROTOCOL,
            expected_component="detector",
            expected_signature=resume_signature,
            expected_total_epochs=epochs,
            model=model,
            optimizer=optimizer,
        )
        start_epoch = int(restored["completed_epoch"]) + 1
        history = list(restored["history"])
        best_state = dict(restored["best_state"])
        best_metric = float(restored["best_metric"])
        elapsed_before = float(restored["elapsed_seconds"])
    for epoch in range(start_epoch, epochs + 1):
        frozen = not smoke and epoch <= args.detector_frozen_epochs
        model.set_early_encoder_frozen(frozen)
        train.set_epoch(epoch)
        loader = _loader(
            train,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed + 10_000 + epoch,
            workers=0 if smoke else args.workers,
            collate_fn=detection_collate,
        )
        model.train()
        loss_sum = samples = 0
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            logits = model(image)
            loss = text_detection_loss(logits, target)
            _require(bool(torch.isfinite(loss)), "detector loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * image.shape[0]
            samples += image.shape[0]
        calibration_metrics = evaluate_detector(
            model,
            calibration,
            device=device,
            batch_size=batch_size,
            workers=0 if smoke else args.workers,
        )
        row = {
            "epoch": epoch,
            "early_encoder_frozen": frozen,
            "train_loss": loss_sum / samples,
            "train_samples": samples,
            "calibration": calibration_metrics,
        }
        history.append(row)
        metric = float(calibration_metrics["pixel_dice_at_0_40"])
        if metric > best_metric:
            best_metric = metric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if not smoke:
            _require(resume_path is not None, "formal detector journal path is missing")
            save_epoch_journal(
                resume_path,
                protocol=DETECTOR_RESUME_PROTOCOL,
                component="detector",
                signature=resume_signature,
                completed_epoch=epoch,
                total_epochs=epochs,
                model=model,
                optimizer=optimizer,
                history=history,
                best_state=best_state,
                best_metric=best_metric,
                elapsed_seconds=elapsed_before + time.perf_counter() - attempt_started,
            )
        print(
            f"numeric-ocr detector seed={args.seed} epoch={epoch}/{epochs} "
            f"loss={row['train_loss']:.5f} cal_dice={metric:.4f}",
            flush=True,
        )
    _require(best_state is not None, "detector produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    validation_metrics = evaluate_detector(
        model,
        validation,
        device=device,
        batch_size=batch_size,
        workers=0 if smoke else args.workers,
    )
    return model, {
        "history": history,
        "selection": "maximum calibration pixel Dice at 0.40; first tie",
        "best_calibration_pixel_dice": best_metric,
        "validation": validation_metrics,
        "train_samples": len(train),
        "calibration_samples": len(calibration),
        "validation_samples": len(validation),
        "image_size": image_size,
        "pretrained_identity": model.pretrained_identity,
        "early_encoder_frozen_epochs": 0 if smoke else args.detector_frozen_epochs,
        "resume": {
            "journal": None if smoke else str(resume_path),
            "signature_sha256": None if smoke else resume_signature_sha256(resume_signature),
            "resumed_from_epoch": start_epoch - 1,
        },
    }


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(
    *,
    component: str,
    state_dict: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    corpus: OCRCorpus,
    metrics: Mapping[str, Any],
    model_config: Mapping[str, Any],
    mode: str,
) -> Mapping[str, Any]:
    return {
        "protocol": CHECKPOINT_PROTOCOL,
        "status": "complete" if mode == "formal" else "smoke_only",
        "component": component,
        "mode": mode,
        "seed": int(args.seed),
        "vocabulary": list(VOCABULARY),
        "prediction_space": "variable-length signed real numeric strings",
        "model_config": dict(model_config),
        "corpus": {
            "root": str(corpus.root),
            "summary_sha256": sha256_file(corpus.root / "summary.json"),
            "samples_sha256": corpus.summary["artifacts"]["samples_sha256"],
            "tokens_sha256": corpus.summary["artifacts"]["tokens_sha256"],
        },
        "metrics": dict(metrics),
        "state_dict": {key: value.detach().cpu() for key, value in state_dict.items()},
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = Path(args.output_dir).resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"training output must stay below {SAFE_OUTPUT_ROOT}") from error
    _require(output != SAFE_OUTPUT_ROOT, "refusing broad training output root")
    corpus = OCRCorpus.load(args.corpus)
    if args.validate_only:
        return {
            "protocol": PROTOCOL,
            "status": "validated",
            "corpus": str(corpus.root),
            "samples": len(corpus.samples),
            "tokens": len(corpus.tokens),
            "group_disjoint_split_sha256": canonical_sha256(corpus.summary["split"]),
            "gpu_work_started": False,
        }

    smoke = bool(args.smoke)
    if smoke:
        _require(args.device == "cpu", "smoke mode must run on CPU")
    resume = bool(getattr(args, "resume", False))
    _require(not resume or bool(args.run_formal), "resume is allowed only for a formal run")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    seed_everything(args.seed)
    run_dir = output / ("smoke" if smoke else f"seed_{args.seed}")
    summary_path = run_dir / "summary.json"
    if not smoke:
        _require(not summary_path.exists(), "completed formal summary exists; refusing overwrite")
        if resume:
            _require(run_dir.is_dir(), "resume requires an existing incomplete run directory")
        else:
            _require(not run_dir.exists(), "formal run directory exists; pass --resume only if incomplete")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    components: dict[str, Any] = {}
    artifact_paths: dict[str, str] = {}
    reused_components: list[str] = []

    if args.component in ("both", "recognizer"):
        path = run_dir / ("recognizer_smoke.pt" if smoke else "recognizer.pt")
        if resume and path.is_file():
            checkpoint = _load_completed_component(
                path, component="recognizer", args=args, corpus=corpus
            )
            metrics = checkpoint["metrics"]
            reused_components.append("recognizer")
        else:
            _require(not path.exists(), "refusing to overwrite recognizer checkpoint")
            recognizer, metrics = train_recognizer(
                corpus, args=args, device=device, smoke=smoke
            )
            checkpoint = _checkpoint(
                component="recognizer",
                state_dict=recognizer.state_dict(),
                args=args,
                corpus=corpus,
                metrics=metrics,
                model_config={"height": 32, "width": 160, "architecture": "TinyCTCRecognizer"},
                mode="smoke" if smoke else "formal",
            )
            _atomic_torch(path, checkpoint)
        components["recognizer"] = metrics
        artifact_paths["recognizer"] = str(path)

    if args.component in ("both", "detector"):
        path = run_dir / ("detector_smoke.pt" if smoke else "detector.pt")
        journal_path = None if smoke else run_dir / "detector.last.pt"
        if resume and path.is_file():
            checkpoint = _load_completed_component(
                path, component="detector", args=args, corpus=corpus
            )
            metrics = checkpoint["metrics"]
            reused_components.append("detector")
        else:
            _require(not path.exists(), "refusing to overwrite detector checkpoint")
            detector, metrics = train_detector(
                corpus,
                args=args,
                device=device,
                smoke=smoke,
                resume_path=journal_path,
            )
            checkpoint = _checkpoint(
                component="detector",
                state_dict=detector.state_dict(),
                args=args,
                corpus=corpus,
                metrics=metrics,
                model_config={
                    "image_size": metrics["image_size"],
                    "architecture": "MobileNetV3SmallFPN",
                    "pretrained_identity": metrics["pretrained_identity"],
                    "annular_geometry_channels": 0,
                    "annular_geometry_extension_ready": True,
                },
                mode="smoke" if smoke else "formal",
            )
            _atomic_torch(path, checkpoint)
        components["detector"] = metrics
        artifact_paths["detector"] = str(path)

    if not smoke:
        _require(not summary_path.exists(), "completed formal summary appeared; refusing overwrite")
    summary = {
        "protocol": PROTOCOL,
        "status": "smoke_complete" if smoke else "complete",
        "mode": "smoke" if smoke else "formal",
        "seed": int(args.seed),
        "device": str(device),
        "corpus": _corpus_identity(corpus),
        "component_selection": args.component,
        "elapsed_seconds": time.perf_counter() - started,
        "components": components,
        "artifacts": artifact_paths,
        "artifact_sha256": {
            name: sha256_file(Path(path)) for name, path in artifact_paths.items()
        },
        "resume": {
            "requested": resume,
            "reused_completed_components": reused_components,
            "detector_epoch_journal": (
                None if smoke or args.component not in ("both", "detector")
                else str(run_dir / "detector.last.pt")
            ),
        },
        "gpu_work_started": device.type == "cuda",
    }
    _atomic_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
