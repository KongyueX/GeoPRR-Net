"""Independently verify a formal VDN phase-2 convergence continuation."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from experiments.freeze_vdn_phase2_syncg_train_inventory import (
    DEFAULT_OUTPUT as DEFAULT_PHASE2_CONTENT_INVENTORY,
    SYNCG_TRAIN_EXPECTED_ROWS,
    verify_inventory_artifact,
)
from experiments.probe_vdn_phase2_determinism import (
    validate_authorized_runtime_environment,
    validate_determinism_authorization_report,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    build_vdn_model,
    sha256_file,
    sha256_source_file,
    verify_vdn_source,
)
from experiments.vdn_phase2_protocol import (
    PHASE2_CHECKPOINT_PROTOCOL,
    PHASE2_END_EPOCH,
    PHASE2_PROTOCOL,
    PHASE2_SCHEMA_VERSION,
    PHASE2_SOURCE_HASH_PROTOCOL,
    PHASE2_VERIFICATION_PROTOCOL,
    apply_phase2_determinism_policy,
    build_phase2_signature,
    convergence_diagnostics,
    current_runtime_environment,
    expected_optimizer_steps,
    formal_phase2_seed,
    load_parent_lineage,
    nested_state_equal,
    normalize_content_inventory_identity,
    strict_model_state_health,
    validate_authoritative_checkpoint,
    validate_parent_training_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--parent-run",
        type=Path,
        help="defaults to the immutable absolute parent path signed by phase 2",
    )
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
        default=DEFAULT_PHASE2_CONTENT_INVENTORY,
        help="frozen SyncG-train inventory that will be freshly rehashed",
    )
    parser.add_argument(
        "--determinism-report",
        type=Path,
        required=True,
        help="the same passed v2 authorization report signed by training",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="must reproduce the CUDA runtime signed by the trainer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new verification report path; existing files are never overwritten",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="retain a negative report without raising when convergence fails",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_json_no_clobber(
    value: Mapping[str, Any],
    output: Path,
) -> Path:
    """Atomically publish JSON through a same-directory hard link."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite verification report: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite verification report: {output}"
            ) from exc
        return output
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def _exclusive_verification_lock(run_dir: Path) -> Iterator[None]:
    """Prevent a trainer from mutating the run during independent verification."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    lock_path = run_dir / "writer.lock"
    payload = json.dumps(
        {
            "attempt": uuid.uuid4().hex,
            "pid": os.getpid(),
            "protocol": "vdn_phase2_exclusive_verification_lock_v1",
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
            f"VDN phase-2 run already has a writer/verification lock: "
            f"{lock_path}"
        ) from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            existing = lock_path.read_bytes()
        except FileNotFoundError:
            existing = None
        if existing == payload:
            lock_path.unlink()


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def _current_training_source_hashes(vdn_source: Path) -> dict[str, str]:
    return {
        "phase2_trainer": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_vdn_phase2.py"
        ),
        "phase2_protocol": sha256_source_file(
            PROJECT_DIR / "experiments" / "vdn_phase2_protocol.py"
        ),
        "base_trainer": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_vdn_syncg.py"
        ),
        "adapter": sha256_source_file(
            PROJECT_DIR / "experiments" / "vdn_baseline.py"
        ),
        "vdn_model": sha256_source_file(
            Path(vdn_source) / "libs" / "models" / "vdn_model.py"
        ),
    }


def _fresh_content_inventory_identity(
    *,
    report_path: Path,
    manifest: Path,
) -> dict[str, Any]:
    inventory_tool = (
        PROJECT_DIR
        / "experiments"
        / "freeze_vdn_phase2_syncg_train_inventory.py"
    )
    verification = verify_inventory_artifact(
        report_path,
        project_root=PROJECT_DIR,
        manifest=manifest,
        vdn_protocol_document=(
            PROJECT_DIR / "docs" / "VDN_CONVERGENCE_PROTOCOL_CN.md"
        ),
        vdn_protocol_source=(
            PROJECT_DIR / "experiments" / "vdn_phase2_protocol.py"
        ),
        inventory_tool_source=inventory_tool,
        expected_rows=SYNCG_TRAIN_EXPECTED_ROWS,
        formal_identity=True,
        workers=1,
    )
    return normalize_content_inventory_identity(
        verification,
        inventory_tool_source=inventory_tool,
        manifest=manifest,
    )


def _expected_summary(
    authoritative: dict[str, Any],
    *,
    lineage,
) -> dict[str, Any]:
    convergence = convergence_diagnostics(
        lineage.summary["history"],
        authoritative["history"],
    )
    return {
        "protocol": authoritative["signature"]["protocol"],
        "schema_version": PHASE2_SCHEMA_VERSION,
        "status": "complete",
        "signature": authoritative["signature"],
        "authoritative_checkpoint_protocol": PHASE2_CHECKPOINT_PROTOCOL,
        "authoritative_epoch": PHASE2_END_EPOCH,
        "best_epoch": int(authoritative["best_epoch"]),
        "best_validation_angle_mae_degrees": float(
            authoritative["best_angle"]
        ),
        "best_origin": authoritative["best_origin"],
        "parent_best_epoch": int(lineage.summary["best_epoch"]),
        "parent_best_validation_angle_mae_degrees": float(
            lineage.summary["best_validation_angle_mae_degrees"]
        ),
        "history": authoritative["history"],
        "convergence": convergence,
        "environment": authoritative["environment"],
    }


def _expected_best_artifact(
    authoritative: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "checkpoint_protocol": PHASE2_CHECKPOINT_PROTOCOL,
        "signature": authoritative["signature"],
        "epoch": int(authoritative["best_epoch"]),
        "best_angle": float(authoritative["best_angle"]),
        "best_origin": authoritative["best_origin"],
        "authoritative_epoch": int(authoritative["epoch"]),
        "model_state": authoritative["best_model_state"],
    }


def _validate_derived_best(
    best: dict[str, Any],
    *,
    authoritative: dict[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    if not isinstance(best, dict):
        raise ValueError("VDN phase-2 best checkpoint is not a dictionary")
    expected = _expected_best_artifact(authoritative)
    _require_equal(
        set(best),
        set(expected),
        label="phase derived-best schema",
    )
    for key in sorted(set(expected) - {"model_state"}):
        _require_equal(
            best[key],
            expected[key],
            label=f"phase derived-best {key}",
        )
    if not nested_state_equal(best["model_state"], expected["model_state"]):
        raise ValueError(
            "phase derived-best model differs from authoritative embedded best"
        )
    return strict_model_state_health(
        model,
        best["model_state"],
        label="phase derived-best model state",
    )


def _verify_phase2_run_locked(
    run_dir: Path,
    *,
    parent_run: Path | None,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path = DEFAULT_PHASE2_CONTENT_INVENTORY,
    determinism_report: Path,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest = Path(manifest).resolve()
    vdn_source = Path(vdn_source).resolve()
    content_inventory = Path(content_inventory).resolve()
    determinism_report = Path(determinism_report).resolve()
    device = torch.device(device)
    paths = {
        "summary": run_dir / "summary.json",
        "best": run_dir / "best.pt",
        "last": run_dir / "last.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete VDN phase-2 run: {missing}")
    if device.type != "cuda":
        raise ValueError("formal VDN phase-2 verification requires CUDA")
    determinism_authorization = validate_determinism_authorization_report(
        determinism_report,
        manifest=manifest,
        vdn_source=vdn_source,
        content_inventory_path=content_inventory,
    )
    apply_phase2_determinism_policy()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    summary = _read_json(paths["summary"])
    signature = summary.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("VDN phase-2 summary signature is invalid")
    signed_parent = signature.get("parent")
    if not isinstance(signed_parent, dict):
        raise ValueError("VDN phase-2 signed parent is invalid")
    signed_parent_value = signed_parent.get("run_dir")
    if not isinstance(signed_parent_value, str) or not signed_parent_value:
        raise ValueError("VDN phase-2 signed parent path is invalid")
    signed_parent_dir = Path(signed_parent_value)
    if not signed_parent_dir.is_absolute():
        raise ValueError("VDN phase-2 signed parent path is not absolute")
    signed_parent_dir = signed_parent_dir.resolve()
    if parent_run is not None:
        _require_equal(
            Path(parent_run).resolve(),
            signed_parent_dir,
            label="requested parent run",
        )

    content_inventory_identity = _fresh_content_inventory_identity(
        report_path=content_inventory,
        manifest=manifest,
    )
    lineage = load_parent_lineage(
        signed_parent_dir,
        manifest=manifest,
        load_checkpoints=True,
    )
    model = build_vdn_model(
        vdn_source,
        image_size=int(lineage.summary["signature"]["image_size"]),
        imagenet_pretrained=False,
    )
    parent_health = validate_parent_training_state(lineage, model)
    runtime_environment = current_runtime_environment(device)
    runtime_authorization = validate_authorized_runtime_environment(
        determinism_authorization,
        runtime_environment,
    )
    runtime_environment["determinism_authorization"] = runtime_authorization
    expected_signature = build_phase2_signature(
        lineage,
        manifest=manifest,
        vdn_source_commit=verify_vdn_source(vdn_source),
        source_sha256=_current_training_source_hashes(vdn_source),
        runtime_environment=runtime_environment,
        parent_optimizer_step=int(parent_health["optimizer_step"]),
        content_inventory_identity=content_inventory_identity,
        determinism_authorization=determinism_authorization,
    )
    _require_equal(
        signature,
        expected_signature,
        label="complete independently reconstructed phase-2 signature",
    )

    authoritative = torch.load(
        paths["last"],
        map_location="cpu",
        weights_only=False,
    )
    authoritative_health = validate_authoritative_checkpoint(
        authoritative,
        lineage=lineage,
        model=model,
        expected_signature=expected_signature,
        parent_health=parent_health,
    )
    _require_equal(
        int(authoritative_health["epoch"]),
        PHASE2_END_EPOCH,
        label="phase authoritative terminal epoch",
    )
    expected_summary = _expected_summary(
        authoritative,
        lineage=lineage,
    )
    _require_equal(
        summary,
        expected_summary,
        label="complete phase summary reconstructed from authoritative state",
    )

    best = torch.load(
        paths["best"],
        map_location="cpu",
        weights_only=False,
    )
    best_health = _validate_derived_best(
        best,
        authoritative=authoritative,
        model=model,
    )
    convergence = expected_summary["convergence"]
    per_epoch_attempts = expected_optimizer_steps(
        int(expected_signature["train_samples"]),
        int(expected_signature["batch_size"]),
    )
    phase_history_health = authoritative_health["history"]
    attempted_steps = int(
        phase_history_health["cumulative_attempted_optimizer_steps"]
    )
    successful_steps = int(
        phase_history_health["cumulative_optimizer_steps"]
    )
    skipped_steps = int(
        phase_history_health["cumulative_skipped_optimizer_steps"]
    )
    return {
        "protocol": PHASE2_VERIFICATION_PROTOCOL,
        "verified": True,
        "training_artifacts_verified": True,
        "eligible_for_test_evaluation": bool(convergence["passed"]),
        "run_dir": str(run_dir),
        "parent_run_dir": str(lineage.run_dir),
        "parent_summary_sha256": sha256_file(lineage.summary_path),
        "parent_last_checkpoint_sha256": sha256_file(lineage.last_path),
        "parent_best_checkpoint_sha256": sha256_file(lineage.best_path),
        "parent_verification_sha256": sha256_file(
            lineage.verification_path
        ),
        "phase_seed": formal_phase2_seed(lineage.seed),
        "epochs": PHASE2_END_EPOCH,
        "best_epoch": int(authoritative["best_epoch"]),
        "best_validation_angle_mae_degrees": float(
            authoritative["best_angle"]
        ),
        "convergence": convergence,
        "optimizer_attempts_per_epoch": per_epoch_attempts,
        "optimizer_attempts": attempted_steps,
        "optimizer_steps": successful_steps,
        "skipped_optimizer_steps": skipped_steps,
        "skipped_optimizer_step_rate": (
            skipped_steps / attempted_steps
        ),
        "max_skipped_optimizer_step_rate": expected_signature[
            "max_skipped_step_rate"
        ],
        "full_phase_max_skipped_optimizer_steps": expected_signature[
            "full_phase_max_skipped_steps"
        ],
        "content_inventory": content_inventory_identity,
        "determinism_authorization": determinism_authorization,
        "determinism_policy": expected_signature["determinism"],
        "authorized_runtime_environment": expected_signature[
            "runtime_environment"
        ],
        "best_checkpoint_sha256": sha256_file(paths["best"]),
        "last_checkpoint_sha256": sha256_file(paths["last"]),
        "summary_sha256": sha256_file(paths["summary"]),
        "source_hash_protocol": PHASE2_SOURCE_HASH_PROTOCOL,
        "verifier_source_sha256": sha256_source_file(
            Path(__file__).resolve()
        ),
        "parent_state_health": parent_health,
        "authoritative_checkpoint_health": authoritative_health,
        "derived_best_model_health": best_health,
    }


def verify_phase2_run(
    run_dir: Path,
    *,
    parent_run: Path | None,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path = DEFAULT_PHASE2_CONTENT_INVENTORY,
    determinism_report: Path,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    with _exclusive_verification_lock(run_dir):
        return _verify_phase2_run_locked(
            run_dir,
            parent_run=parent_run,
            manifest=manifest,
            vdn_source=vdn_source,
            content_inventory=content_inventory,
            determinism_report=determinism_report,
            device=device,
        )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite verification report: {output}"
        )
    run_dir = args.run_dir.resolve()
    with _exclusive_verification_lock(run_dir):
        result = _verify_phase2_run_locked(
            run_dir,
            parent_run=args.parent_run,
            manifest=args.manifest,
            vdn_source=args.vdn_source,
            content_inventory=args.content_inventory,
            determinism_report=args.determinism_report,
            device=args.device,
        )
        write_json_no_clobber(result, output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)
    if (
        not args.report_only
        and result["eligible_for_test_evaluation"] is not True
    ):
        raise RuntimeError(
            "VDN phase-2 training artifacts are valid, but the frozen "
            "convergence gate failed; test evaluation remains blocked"
        )


if __name__ == "__main__":
    main()
