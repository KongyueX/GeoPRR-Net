"""Train/smoke the independent Mobile-SVTR CTC fallback on public SyncG only."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.syncg_numeric_ocr import (
    BLANK_INDEX,
    PROTOCOL,
    VOCABULARY,
    OCRCorpus,
    SyncGTextRecognitionImageDataset,
    canonical_sha256,
    recognition_image_collate,
    sha256_file,
)
from experiments.syncg_strong_numeric_ocr import (
    DEFAULT_OFFICIAL_WEIGHTS,
    STRONG_ARCHITECTURE,
    STRONG_CHECKPOINT_PROTOCOL,
    MobileSVTRCTCRecognizer,
    model_inventory,
    verify_official_backbone,
)
from experiments.train_syncg_numeric_ocr import evaluate_recognizer
from experiments.ocr_training_resume import (
    canonical_sha256 as resume_signature_sha256,
    load_epoch_journal,
    save_epoch_journal,
)


SAFE_OUTPUT_ROOT: Final[Path] = Path(r"C:\pointer_read").resolve()
DEFAULT_CORPUS: Final[Path] = SAFE_OUTPUT_ROOT / "syncg_numeric_ocr_public_v1"
DEFAULT_OUTPUT: Final[Path] = SAFE_OUTPUT_ROOT / "syncg_strong_numeric_ocr_runs"
IMPLEMENTATION_PATH: Final[Path] = PROJECT_ROOT / "experiments/syncg_strong_numeric_ocr.py"
TRAINER_PATH: Final[Path] = Path(__file__).resolve()
EVALUATOR_PATH: Final[Path] = PROJECT_ROOT / "experiments/train_syncg_numeric_ocr.py"
RESUME_UTILITY_PATH: Final[Path] = PROJECT_ROOT / "experiments/ocr_training_resume.py"
STRONG_RESUME_PROTOCOL: Final[str] = "syncg_public_strong_numeric_ocr_resume_v1"

# Frozen before the Tiny recognizer's independent validation was observed.
# The component and end-to-end gates are intentionally separate: poor GARC
# coverage with a strong recognizer points to detector recall, not to another
# recognizer iteration.
ACTIVATION_GATE: Final[dict[str, Any]] = {
    "gate_version": "tiny_to_strong_ocr_v1",
    "activate_strong_recognizer_if_any": {
        "tiny_validation_exact_accuracy_below": 0.92,
        "tiny_validation_character_accuracy_below": 0.98,
        "tiny_validation_parseable_fraction_below": 0.995,
        "garc_conditional_pair_exact_below": 0.90,
    },
    "activate_dbnet_plus_plus_detector_if_any": {
        "garc_full_denominator_coverage_below": 0.90,
        "detector_box_recall_below": 0.92,
    },
    "retain_strong_recognizer_if": {
        "validation_exact_accuracy_gain_min": 0.02,
        "or_garc_full_denominator_pair_exact_gain_min": 0.03,
        "maximum_garc_coverage_regression": 0.01,
    },
    "selection_data": "calibration physical groups only",
    "validation_use": "one-shot reporting after checkpoint selection",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained-backbone", type=Path, default=DEFAULT_OFFICIAL_WEIGHTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--frozen-epochs", type=int, default=2)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--decimal-probability", type=float, default=0.20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only an incomplete formal run from its authenticated last epoch",
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
    dataset: SyncGTextRecognitionImageDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
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
        collate_fn=recognition_image_collate,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _corpus_identity(corpus: OCRCorpus) -> dict[str, Any]:
    return {
        "root": str(corpus.root),
        "summary_sha256": sha256_file(corpus.root / "summary.json"),
        "samples_sha256": corpus.summary["artifacts"]["samples_sha256"],
        "tokens_sha256": corpus.summary["artifacts"]["tokens_sha256"],
    }


def _strong_resume_signature(
    *,
    args: argparse.Namespace,
    corpus: OCRCorpus,
    initialization: Mapping[str, Any],
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": STRONG_RESUME_PROTOCOL,
        "component": "recognizer",
        "seed": int(args.seed),
        "corpus": _corpus_identity(corpus),
        "configuration": {
            "epochs": int(args.epochs),
            "frozen_epochs": int(args.frozen_epochs),
            "image_batch_size": int(args.image_batch_size),
            "learning_rate": float(args.learning_rate),
            "decimal_probability": float(args.decimal_probability),
            "architecture": STRONG_ARCHITECTURE,
            "embedding_dim": 256,
            "attention_heads": 8,
            "attention_layers": 4,
            "feedforward_dim": 768,
            "dropout": 0.10,
        },
        "initialization": dict(initialization),
        "activation_gate_sha256": canonical_sha256(ACTIVATION_GATE),
        "code": {
            **dict(code_identity),
            "resume_utility": str(RESUME_UTILITY_PATH),
            "resume_utility_sha256": sha256_file(RESUME_UTILITY_PATH),
        },
    }


def _load_completed_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    corpus: OCRCorpus,
    initialization: Mapping[str, Any],
    code_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    _require(isinstance(value, Mapping), "completed strong checkpoint is not a mapping")
    _require(value.get("protocol") == STRONG_CHECKPOINT_PROTOCOL, "completed strong protocol drift")
    _require(value.get("status") == "complete", "completed strong checkpoint is not formal")
    _require(value.get("component") == "recognizer", "completed strong component drift")
    _require(value.get("mode") == "formal", "completed strong mode drift")
    _require(int(value.get("seed", -1)) == int(args.seed), "completed strong seed drift")
    _require(value.get("vocabulary") == list(VOCABULARY), "completed strong vocabulary drift")
    _require(value.get("corpus") == _corpus_identity(corpus), "completed strong corpus drift")
    _require(value.get("initialization") == dict(initialization), "completed strong initialization drift")
    _require(value.get("code") == dict(code_identity), "completed strong code identity drift")
    _require(
        value.get("activation_gate_sha256") == canonical_sha256(ACTIVATION_GATE),
        "completed strong activation gate drift",
    )
    _require(isinstance(value.get("metrics"), Mapping), "completed strong metrics missing")
    _require(isinstance(value.get("state_dict"), Mapping), "completed strong state missing")
    return value


def train(
    corpus: OCRCorpus,
    *,
    args: argparse.Namespace,
    device: torch.device,
    smoke: bool,
    initialization: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    resume_path: Path | None = None,
) -> tuple[MobileSVTRCTCRecognizer, Mapping[str, Any]]:
    limit = 2 if smoke else None
    train_set = SyncGTextRecognitionImageDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="train",
        seed=args.seed,
        training=True,
        decimal_probability=args.decimal_probability,
        limit=limit,
    )
    calibration_set = SyncGTextRecognitionImageDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="calibration",
        seed=args.seed,
        training=False,
        limit=limit,
    )
    validation_set = SyncGTextRecognitionImageDataset(
        corpus,
        project_root=PROJECT_ROOT,
        partition="validation",
        seed=args.seed,
        training=False,
        limit=limit,
    )
    model = MobileSVTRCTCRecognizer(
        pretrained_backbone_path=Path(initialization["path"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    epochs = 1 if smoke else int(args.epochs)
    image_batch_size = max(1, min(int(args.image_batch_size), len(train_set)))
    best_state: dict[str, torch.Tensor] | None = None
    best_metric = -math.inf
    history: list[dict[str, Any]] = []
    start_epoch = 1
    elapsed_before = 0.0
    attempt_started = time.perf_counter()
    resume_signature = _strong_resume_signature(
        args=args,
        corpus=corpus,
        initialization=initialization,
        code_identity=code_identity,
    )
    if resume_path is not None and resume_path.is_file():
        _require(not smoke, "strong smoke run cannot resume")
        restored = load_epoch_journal(
            resume_path,
            expected_protocol=STRONG_RESUME_PROTOCOL,
            expected_component="recognizer",
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
        train_set.set_epoch(epoch)
        frozen = epoch <= int(args.frozen_epochs) and not smoke
        model.set_backbone_frozen(frozen)
        loader = _loader(
            train_set,
            batch_size=image_batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            workers=0 if smoke else int(args.workers),
        )
        model.train()
        loss_sum = samples = synthesized = 0
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            logits = model(images)
            input_lengths = torch.full(
                (images.shape[0],), logits.shape[0], dtype=torch.long, device=device
            )
            _require(
                bool(torch.all(target_lengths <= input_lengths)),
                "strong CTC time axis is shorter than a target",
            )
            loss = criterion(
                logits.log_softmax(2), targets, input_lengths, target_lengths
            )
            _require(bool(torch.isfinite(loss)), "strong recognizer loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * images.shape[0]
            samples += images.shape[0]
            synthesized += int(batch["decimal_synthesized"].sum())
        calibration = evaluate_recognizer(
            model,
            calibration_set,
            device=device,
            batch_size=image_batch_size,
            workers=0 if smoke else int(args.workers),
        )
        metric = float(calibration["exact_accuracy"])
        row = {
            "epoch": epoch,
            "backbone_frozen": frozen,
            "train_loss": loss_sum / samples,
            "train_tokens": samples,
            "decimal_synthesized_fraction": synthesized / samples,
            "calibration": calibration,
        }
        history.append(row)
        if metric > best_metric:
            best_metric = metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if not smoke:
            _require(resume_path is not None, "formal strong journal path is missing")
            save_epoch_journal(
                resume_path,
                protocol=STRONG_RESUME_PROTOCOL,
                component="recognizer",
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
            f"strong-numeric-ocr seed={args.seed} epoch={epoch}/{epochs} "
            f"loss={row['train_loss']:.5f} cal_exact={metric:.4f}",
            flush=True,
        )
    _require(best_state is not None, "strong recognizer produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    validation = evaluate_recognizer(
        model,
        validation_set,
        device=device,
        batch_size=image_batch_size,
        workers=0 if smoke else int(args.workers),
    )
    return model, {
        "history": history,
        "selection": "maximum calibration exact string accuracy; first tie",
        "best_calibration_exact_accuracy": best_metric,
        "validation": validation,
        "train_source_images": len(train_set),
        "calibration_source_images": len(calibration_set),
        "validation_source_images": len(validation_set),
        "train_tokens": train_set.token_count,
        "calibration_tokens": calibration_set.token_count,
        "validation_tokens": validation_set.token_count,
        "resume": {
            "journal": None if smoke else str(resume_path),
            "signature_sha256": None if smoke else resume_signature_sha256(resume_signature),
            "resumed_from_epoch": start_epoch - 1,
        },
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = Path(args.output_dir).resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"strong OCR output must stay below {SAFE_OUTPUT_ROOT}") from error
    _require(output != SAFE_OUTPUT_ROOT, "refusing broad strong OCR output root")
    corpus = OCRCorpus.load(args.corpus)
    initialization = verify_official_backbone(args.pretrained_backbone)
    code_identity = {
        "implementation": str(IMPLEMENTATION_PATH),
        "implementation_sha256": sha256_file(IMPLEMENTATION_PATH),
        "trainer": str(TRAINER_PATH),
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "evaluator": str(EVALUATOR_PATH),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
    }
    if args.validate_only:
        return {
            "protocol": STRONG_CHECKPOINT_PROTOCOL,
            "status": "validated",
            "corpus": str(corpus.root),
            "corpus_protocol": PROTOCOL,
            "samples": len(corpus.samples),
            "tokens": len(corpus.tokens),
            "group_disjoint_split_sha256": canonical_sha256(corpus.summary["split"]),
            "initialization": initialization,
            "code": code_identity,
            "activation_gate": ACTIVATION_GATE,
            "activation_gate_sha256": canonical_sha256(ACTIVATION_GATE),
            "gpu_work_started": False,
        }

    smoke = bool(args.smoke)
    if smoke:
        _require(args.device == "cpu", "strong OCR smoke mode must run on CPU")
    resume = bool(getattr(args, "resume", False))
    _require(not resume or bool(args.run_formal), "strong resume is allowed only for formal mode")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    seed_everything(int(args.seed))
    run_dir = output / ("smoke" if smoke else f"seed_{args.seed}")
    summary_path = run_dir / "summary.json"
    if not smoke:
        _require(not summary_path.exists(), "completed strong summary exists; refusing overwrite")
        if resume:
            _require(run_dir.is_dir(), "strong resume requires an existing incomplete run directory")
        else:
            _require(not run_dir.exists(), "strong run directory exists; pass --resume only if incomplete")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mode = "smoke" if smoke else "formal"
    checkpoint_path = run_dir / (
        "recognizer_smoke.pt" if smoke else "recognizer.pt"
    )
    reused_completed_checkpoint = bool(resume and checkpoint_path.is_file())
    if reused_completed_checkpoint:
        checkpoint = _load_completed_checkpoint(
            checkpoint_path,
            args=args,
            corpus=corpus,
            initialization=initialization,
            code_identity=code_identity,
        )
        metrics = checkpoint["metrics"]
        config = checkpoint["model_config"]
        model = MobileSVTRCTCRecognizer(
            embedding_dim=int(config["embedding_dim"]),
            attention_heads=int(config["attention_heads"]),
            attention_layers=int(config["attention_layers"]),
            feedforward_dim=int(config["feedforward_dim"]),
            dropout=float(config["dropout"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        _require(not checkpoint_path.exists(), "refusing to overwrite strong checkpoint")
        journal_path = None if smoke else run_dir / "recognizer.last.pt"
        model, metrics = train(
            corpus,
            args=args,
            device=device,
            smoke=smoke,
            initialization=initialization,
            code_identity=code_identity,
            resume_path=journal_path,
        )
        checkpoint = {
            "protocol": STRONG_CHECKPOINT_PROTOCOL,
            "status": "smoke_only" if smoke else "complete",
            "component": "recognizer",
            "mode": mode,
            "seed": int(args.seed),
            "vocabulary": list(VOCABULARY),
            "prediction_space": "variable-length signed real numeric strings",
            "model_config": {
                "height": 32,
                "width": 160,
                "internal_height": 48,
                "internal_width": 256,
                "ctc_time_steps": 16,
                "architecture": STRONG_ARCHITECTURE,
                "embedding_dim": 256,
                "attention_heads": 8,
                "attention_layers": 4,
                "feedforward_dim": 768,
                "dropout": 0.10,
                **model_inventory(model),
            },
            "initialization": dict(initialization),
            "code": code_identity,
            "corpus": _corpus_identity(corpus),
            "activation_gate": ACTIVATION_GATE,
            "activation_gate_sha256": canonical_sha256(ACTIVATION_GATE),
            "metrics": metrics,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        }
        _atomic_torch(checkpoint_path, checkpoint)
    if not smoke:
        _require(not summary_path.exists(), "completed strong summary appeared; refusing overwrite")
    summary = {
        "protocol": STRONG_CHECKPOINT_PROTOCOL,
        "status": "smoke_complete" if smoke else "complete",
        "mode": mode,
        "seed": int(args.seed),
        "device": str(device),
        "corpus": _corpus_identity(corpus),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "model_inventory": model_inventory(model),
        "initialization": initialization,
        "code": code_identity,
        "activation_gate": ACTIVATION_GATE,
        "activation_gate_sha256": canonical_sha256(ACTIVATION_GATE),
        "artifact": str(checkpoint_path),
        "artifact_sha256": sha256_file(checkpoint_path),
        "resume": {
            "requested": resume,
            "reused_completed_checkpoint": reused_completed_checkpoint,
            "epoch_journal": None if smoke else str(run_dir / "recognizer.last.pt"),
        },
        "gpu_work_started": device.type == "cuda",
    }
    _atomic_json(summary_path, summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
