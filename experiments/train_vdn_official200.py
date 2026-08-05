"""Train one frozen from-scratch VDN run to the official 200-epoch boundary."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.utils.data import DataLoader

from experiments.preflight_vdn_official200 import (
    DEFAULT_OUTPUT as DEFAULT_PREFLIGHT,
    DEFAULT_RUN_ROOT,
    _fresh_content_inventory_identity,
    _strict_json,
    validate_preflight_report,
)
from experiments.train_vdn_syncg import (
    _json_write,
    _torch_save,
    _train_epoch,
    _validate,
)
from experiments.vdn_baseline import (
    SyncGVDNDataset,
    build_vdn_model,
    grouped_train_val_split,
    load_syncg_manifest,
    seed_worker,
    set_random_seed,
    verify_vdn_source,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_BATCH_SIZE,
    OFFICIAL200_CHECKPOINT_PROTOCOL,
    OFFICIAL200_DETERMINISM_POLICY,
    OFFICIAL200_EPOCHS,
    OFFICIAL200_FORMAL_SEEDS,
    OFFICIAL200_IMAGE_SIZE,
    OFFICIAL200_PROTOCOL,
    OFFICIAL200_ROTATION_FACTOR,
    OFFICIAL200_SCALE_FACTOR,
    OFFICIAL200_SCHEMA_VERSION,
    OFFICIAL200_TRAIN_METRIC_KEYS,
    OFFICIAL200_VALIDATION_FRACTION,
    OFFICIAL200_VALIDATION_METRIC_KEYS,
    OFFICIAL200_WORKERS,
    VDN_TRAINABLE_PARAMETER_TENSORS,
    apply_phase2_determinism_policy,
    assert_syncg_train_manifest_path,
    assert_train_only_path,
    build_official200_signature,
    build_phase2_adam_optimizer,
    current_runtime_environment,
    expected_sample_order_sha256,
    model_state_sha256,
    official200_epoch_seed,
    official200_learning_rate,
    official200_source_hashes,
    official200_vector_weight,
    require_formal_seed,
    tail_diagnostics,
    validate_authorized_runtime_environment,
    validate_authoritative_checkpoint,
    validate_determinism_report_payload,
    validate_live_adam_optimizer,
    validate_official200_history,
    validate_scaler_state,
    validate_scaler_transition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--vdn-source",
        type=Path,
        default=Path("artifacts/vendor/VectorDetectionNetwork"),
    )
    parser.add_argument(
        "--content-inventory",
        type=Path,
        default=Path(
            "artifacts/protocols/"
            "vdn_phase2_syncg_train_content_inventory_v1.json"
        ),
    )
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument(
        "--determinism-report",
        type=Path,
        default=Path(
            "artifacts/protocols/"
            "vdn_official200_determinism_probe_v1.json"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="fault recovery only; the frozen endpoint remains epoch 200",
    )
    return parser.parse_args()


def prepare_output_dir(path: Path, *, resume: bool) -> None:
    path = assert_train_only_path(
        path,
        label="official-200 training output",
    )
    if resume:
        if not path.is_dir():
            raise FileNotFoundError(f"official-200 run is absent: {path}")
        last = path / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(
                f"cannot resume without authoritative checkpoint: {last}"
            )
        if (path / "verification_v1.json").exists():
            raise FileExistsError(
                "verified official-200 runs are immutable and cannot resume"
            )
        return
    if path.exists():
        raise FileExistsError(
            f"refusing to reuse official-200 output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=False)


@contextmanager
def exclusive_writer_lock(output_dir: Path) -> Iterator[str]:
    output_dir = assert_train_only_path(
        output_dir,
        label="official-200 training output",
    )
    lock_path = output_dir / "writer.lock"
    attempt = uuid.uuid4().hex
    payload = json.dumps(
        {
            "attempt": attempt,
            "pid": os.getpid(),
            "protocol": "vdn_official200_exclusive_writer_lock_v1",
        },
        sort_keys=True,
    ).encode("utf-8")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"official-200 output already has a writer lock: {lock_path}"
        ) from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield attempt
    finally:
        os.close(descriptor)
        try:
            existing = lock_path.read_bytes()
        except FileNotFoundError:
            existing = None
        if existing == payload:
            lock_path.unlink()


def _clone_model_state_to_cpu(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def _best_artifact_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": OFFICIAL200_SCHEMA_VERSION,
        "checkpoint_protocol": OFFICIAL200_CHECKPOINT_PROTOCOL,
        "signature": state["signature"],
        "epoch": int(state["best_epoch"]),
        "best_angle": float(state["best_angle"]),
        "authoritative_epoch": int(state["epoch"]),
        "model_state": state["best_model_state"],
    }


def _summary_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    complete = int(state["epoch"]) == OFFICIAL200_EPOCHS
    diagnostic = tail_diagnostics(state["history"]) if complete else None
    return {
        "protocol": OFFICIAL200_PROTOCOL,
        "schema_version": OFFICIAL200_SCHEMA_VERSION,
        "status": "complete" if complete else "running",
        "signature": state["signature"],
        "authoritative_checkpoint_protocol": (
            OFFICIAL200_CHECKPOINT_PROTOCOL
        ),
        "authoritative_epoch": int(state["epoch"]),
        "best_epoch": int(state["best_epoch"]),
        "best_validation_angle_mae_degrees": float(state["best_angle"]),
        "history": state["history"],
        "tail_diagnostic": diagnostic,
        "official_stopping_boundary_reached": complete,
        "additional_training_authorized": False,
        "phase4_authorized": False,
        "environment": state["environment"],
    }


def _recover_derived_artifacts(
    output_dir: Path,
    state: Mapping[str, Any],
) -> None:
    _torch_save(
        Path(output_dir) / "best.pt",
        _best_artifact_from_state(state),
    )
    _json_write(
        Path(output_dir) / "summary.json",
        _summary_from_state(state),
    )


def _train_loader(
    dataset: SyncGVDNDataset,
    *,
    epoch_seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=OFFICIAL200_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(epoch_seed)),
        num_workers=OFFICIAL200_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )


def _validation_loader(
    dataset: SyncGVDNDataset,
    *,
    epoch_seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=OFFICIAL200_BATCH_SIZE,
        shuffle=False,
        generator=torch.Generator().manual_seed(int(epoch_seed)),
        num_workers=OFFICIAL200_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )


def _validate_epoch_metrics(
    *,
    epoch: int,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    expected_train_samples: int,
    expected_validation_samples: int,
    expected_attempts: int,
    expected_order: str,
    prior_scaler_state: Mapping[str, Any],
    scaler_state: Mapping[str, Any],
) -> tuple[int, int]:
    if set(train_metrics) != OFFICIAL200_TRAIN_METRIC_KEYS:
        raise RuntimeError(
            f"official-200 epoch {epoch} train metric schema drifted"
        )
    if set(validation_metrics) != OFFICIAL200_VALIDATION_METRIC_KEYS:
        raise RuntimeError(
            f"official-200 epoch {epoch} validation metric schema drifted"
        )
    values = [
        float(value)
        for field, value in train_metrics.items()
        if field
        not in {
            "sample_order_sha256",
            "scaler_start_state",
            "scaler_skipped_batch_indices",
            "scaler_end_state",
        }
    ] + [float(value) for value in validation_metrics.values()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            f"official-200 epoch {epoch} has non-finite metrics"
        )
    successful = int(train_metrics["optimizer_steps"])
    skipped = int(train_metrics["skipped_optimizer_steps"])
    if (
        type(train_metrics["optimizer_steps"]) is not int
        or type(train_metrics["skipped_optimizer_steps"]) is not int
        or successful <= 0
        or skipped < 0
        or successful + skipped != expected_attempts
    ):
        raise RuntimeError(
            f"official-200 epoch {epoch} optimizer accounting is invalid"
        )
    if (
        type(train_metrics["samples"]) is not int
        or int(train_metrics["samples"]) != expected_train_samples
        or type(validation_metrics["samples"]) is not int
        or int(validation_metrics["samples"]) != expected_validation_samples
    ):
        raise RuntimeError(
            f"official-200 epoch {epoch} sample accounting drifted"
        )
    if train_metrics["sample_order_sha256"] != expected_order:
        raise RuntimeError(
            f"official-200 epoch {epoch} sample order drifted"
        )
    if not math.isclose(
        float(train_metrics["vector_weight"]),
        official200_vector_weight(epoch),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            f"official-200 epoch {epoch} vector weight drifted"
        )
    valid_directions = validation_metrics["valid_directions"]
    if (
        type(valid_directions) is not int
        or not 0 < valid_directions <= expected_validation_samples
        or not math.isclose(
            float(validation_metrics["direction_coverage"]),
            valid_directions / expected_validation_samples,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError(
            f"official-200 epoch {epoch} validation coverage drifted"
        )
    if not (
        0.0 <= float(validation_metrics["angle_acc_1deg"])
        <= float(validation_metrics["angle_acc_3deg"])
        <= float(validation_metrics["angle_acc_5deg"])
        <= 1.0
    ):
        raise RuntimeError(
            f"official-200 epoch {epoch} validation accuracy drifted"
        )
    if train_metrics["scaler_start_state"] != dict(prior_scaler_state):
        raise RuntimeError(
            f"official-200 epoch {epoch} scaler continuity drifted"
        )
    transition = validate_scaler_transition(
        train_metrics["scaler_start_state"],
        train_metrics["scaler_end_state"],
        train_metrics["scaler_skipped_batch_indices"],
        attempted_steps=expected_attempts,
        label=f"official-200 epoch {epoch}",
    )
    if (
        transition["successful_steps"] != successful
        or transition["skipped_steps"] != skipped
        or train_metrics["scaler_end_state"] != dict(scaler_state)
    ):
        raise RuntimeError(
            f"official-200 epoch {epoch} scaler trace drifted"
        )
    return successful, skipped


def _run(args: argparse.Namespace, *, preflight: Mapping[str, Any]) -> None:
    seed = require_formal_seed(args.seed)
    manifest = assert_syncg_train_manifest_path(args.manifest)
    vdn_source = args.vdn_source.resolve()
    content_inventory = assert_train_only_path(
        args.content_inventory,
        label="official-200 content inventory",
    )
    run_root = assert_train_only_path(
        args.run_root,
        label="official-200 run root",
    )
    output_dir = assert_train_only_path(
        args.output_dir,
        label="official-200 training output",
    )
    expected_output = run_root / f"seed_{seed}"
    if output_dir != expected_output:
        raise ValueError(
            f"official-200 seed {seed} output must be {expected_output}"
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal official-200 training requires CUDA AMP")
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before official-200 determinism policy"
        )
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError(
            f"PYTHONHASHSEED must equal the formal seed {seed}"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before training"
        )
    if apply_phase2_determinism_policy() != OFFICIAL200_DETERMINISM_POLICY:
        raise RuntimeError("official-200 determinism policy drifted")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    runtime_environment = current_runtime_environment(device)
    runtime_environment["pythonhashseed"] = os.environ["PYTHONHASHSEED"]
    runtime_environment["determinism_authorization"] = (
        validate_authorized_runtime_environment(
            args.determinism_report_data,
            runtime_environment,
        )
    )

    fresh_content = _fresh_content_inventory_identity(
        report_path=content_inventory,
        manifest=manifest,
        workers=1,
    )
    if fresh_content != preflight["content_inventory"]:
        raise RuntimeError(
            "official-200 content identity changed after preflight"
        )
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train_samples, validation_samples = grouped_train_val_split(
        samples,
        validation_fraction=OFFICIAL200_VALIDATION_FRACTION,
        seed=seed,
    )
    run_plan = next(
        run for run in preflight["runs"] if int(run["seed"]) == seed
    )

    set_random_seed(seed)
    model = build_vdn_model(
        vdn_source,
        image_size=OFFICIAL200_IMAGE_SIZE,
        imagenet_pretrained=True,
    )
    initial_model_hash = model_state_sha256(model.state_dict())
    if initial_model_hash != run_plan["initial_model_state_sha256"]:
        raise RuntimeError(
            f"official-200 seed {seed} initialization changed after preflight"
        )
    if verify_vdn_source(vdn_source) != preflight["vdn_source"]["commit"]:
        raise RuntimeError("official-200 pinned VDN commit drifted")
    signature = build_official200_signature(
        seed=seed,
        manifest=manifest,
        manifest_protocol=manifest.with_name(
            manifest.name + ".protocol.json"
        ),
        vdn_source=vdn_source,
        train_samples=train_samples,
        validation_samples=validation_samples,
        initialization_checkpoint=Path(
            preflight["initialization"]["checkpoint_path"]
        ),
        initialization_state_sha256=initial_model_hash,
        content_inventory_identity=fresh_content,
        preflight_binding=args.preflight_binding,
        determinism_authorization=args.determinism_authorization,
        runtime_environment=runtime_environment,
        source_sha256=official200_source_hashes(vdn_source),
    )

    last_path = output_dir / "last.pt"
    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        state_health = validate_authoritative_checkpoint(
            state,
            model=model,
            train_samples=train_samples,
            signature=signature,
        )
        _recover_derived_artifacts(output_dir, state)
        if int(state_health["epoch"]) == OFFICIAL200_EPOCHS:
            print(output_dir / "summary.json")
            print(output_dir / "best.pt")
            return
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_angle = float(state["best_angle"])
        best_model_state = _clone_model_state_to_cpu(
            state["best_model_state"]
        )
        elapsed_offset = float(state["training_elapsed_seconds"])
        model.load_state_dict(state["current_model_state"], strict=True)
    else:
        state = None
        history = []
        start_epoch = 1
        best_epoch = 0
        best_angle = math.inf
        best_model_state = {}
        elapsed_offset = 0.0

    model.to(device)
    optimizer = build_phase2_adam_optimizer(
        model,
        learning_rate=(
            official200_learning_rate(start_epoch - 1)
            if start_epoch > 1
            else official200_learning_rate(1)
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=512.0,
    )
    if state is not None:
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])

    train_dataset = SyncGVDNDataset(
        train_samples,
        image_size=OFFICIAL200_IMAGE_SIZE,
        training=True,
        scale_factor=OFFICIAL200_SCALE_FACTOR,
        rotation_factor=OFFICIAL200_ROTATION_FACTOR,
    )
    validation_dataset = SyncGVDNDataset(
        validation_samples,
        image_size=OFFICIAL200_IMAGE_SIZE,
        training=False,
        scale_factor=OFFICIAL200_SCALE_FACTOR,
        rotation_factor=OFFICIAL200_ROTATION_FACTOR,
    )
    expected_attempts = int(signature["optimizer_attempts_per_epoch"])
    cumulative_successful = sum(
        int(row["train"]["optimizer_steps"]) for row in history
    )
    cumulative_skipped = sum(
        int(row["train"]["skipped_optimizer_steps"]) for row in history
    )
    started = time.perf_counter()

    for epoch in range(start_epoch, OFFICIAL200_EPOCHS + 1):
        epoch_seed = official200_epoch_seed(seed, epoch)
        set_random_seed(epoch_seed)
        learning_rate = official200_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train_loader = _train_loader(
            train_dataset,
            epoch_seed=epoch_seed,
            device=device,
        )
        prior_scaler_state = copy.deepcopy(scaler.state_dict())
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp_enabled=True,
            vector_weight=official200_vector_weight(epoch),
            epoch=epoch,
            record_scaler_trace=True,
        )
        validation_loader = _validation_loader(
            validation_dataset,
            epoch_seed=epoch_seed,
            device=device,
        )
        validation_metrics = _validate(
            model,
            validation_loader,
            device=device,
            amp_enabled=True,
        )
        successful, skipped = _validate_epoch_metrics(
            epoch=epoch,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            expected_train_samples=len(train_samples),
            expected_validation_samples=len(validation_samples),
            expected_attempts=expected_attempts,
            expected_order=expected_sample_order_sha256(
                train_samples,
                epoch_seed=epoch_seed,
            ),
            prior_scaler_state=prior_scaler_state,
            scaler_state=scaler.state_dict(),
        )
        cumulative_successful += successful
        cumulative_skipped += skipped
        if cumulative_skipped > int(
            signature["full_run_max_skipped_optimizer_steps"]
        ):
            raise RuntimeError(
                "official-200 cumulative AMP skips exceed the frozen budget"
            )
        validate_live_adam_optimizer(
            model,
            optimizer,
            expected_step=cumulative_successful,
            expected_learning_rate=learning_rate,
            expected_parameter_count=VDN_TRAINABLE_PARAMETER_TENSORS,
            label=f"official-200 epoch {epoch}",
        )
        validate_scaler_state(
            scaler.state_dict(),
            label=f"official-200 epoch {epoch}",
        )

        current_angle = float(
            validation_metrics["angle_mae_degrees"]
        )
        improved = current_angle < best_angle
        if improved:
            best_epoch = epoch
            best_angle = current_angle
            best_model_state = _clone_model_state_to_cpu(
                model.state_dict()
            )
        elapsed = elapsed_offset + time.perf_counter() - started
        history.append(
            {
                "epoch": epoch,
                "epoch_seed": epoch_seed,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
                "best": improved,
                "training_elapsed_seconds": elapsed,
                "preflight_journal": signature["preflight"],
                "determinism_authorization": signature[
                    "determinism_authorization"
                ],
                "determinism_policy": signature["determinism"],
            }
        )
        complete = epoch == OFFICIAL200_EPOCHS
        if complete:
            validate_official200_history(
                train_samples,
                history,
                through_epoch=OFFICIAL200_EPOCHS,
                signature=signature,
            )
        state = {
            "schema_version": OFFICIAL200_SCHEMA_VERSION,
            "checkpoint_protocol": OFFICIAL200_CHECKPOINT_PROTOCOL,
            "signature": signature,
            "status": "complete" if complete else "running",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_angle": best_angle,
            "current_model_state": _clone_model_state_to_cpu(
                model.state_dict()
            ),
            "best_model_state": best_model_state,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "training_elapsed_seconds": elapsed,
            "environment": runtime_environment,
        }
        _torch_save(last_path, state)
        _recover_derived_artifacts(output_dir, state)
        print(
            f"official200_epoch={epoch}/{OFFICIAL200_EPOCHS} "
            f"lr={learning_rate:g} "
            f"vector_weight={official200_vector_weight(epoch):.8f} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"optimizer_success={successful} "
            f"optimizer_skipped={skipped} "
            f"val_angle_mae={current_angle:.4f}deg "
            f"best={best_angle:.4f}deg"
        )

    print(output_dir / "summary.json")
    print(output_dir / "best.pt")


def main() -> None:
    args = parse_args()
    args.seed = require_formal_seed(args.seed)
    args.manifest = assert_syncg_train_manifest_path(args.manifest)
    args.vdn_source = args.vdn_source.resolve()
    args.content_inventory = assert_train_only_path(
        args.content_inventory,
        label="official-200 content inventory",
    )
    args.preflight = assert_train_only_path(
        args.preflight,
        label="official-200 preflight report",
    )
    args.determinism_report = assert_train_only_path(
        args.determinism_report,
        label="official-200 determinism report",
    )
    args.run_root = assert_train_only_path(
        args.run_root,
        label="official-200 run root",
    )
    args.output_dir = assert_train_only_path(
        args.output_dir,
        label="official-200 training output",
    )
    preflight, binding = validate_preflight_report(
        args.preflight,
        manifest=args.manifest,
        vdn_source=args.vdn_source,
        content_inventory=args.content_inventory,
        run_root=args.run_root,
        require_output_absent=False,
    )
    args.preflight_binding = binding
    args.determinism_report_data = _strict_json(
        args.determinism_report
    )
    args.determinism_authorization = validate_determinism_report_payload(
        args.determinism_report_data,
        report_path=args.determinism_report,
        preflight_binding=binding,
        content_inventory_identity=preflight["content_inventory"],
        vdn_source=args.vdn_source,
    )
    prepare_output_dir(args.output_dir, resume=args.resume)
    with exclusive_writer_lock(args.output_dir):
        _run(args, preflight=preflight)


if __name__ == "__main__":
    main()
