"""Independently verify one completed train-only official-200 VDN run."""
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

from experiments.preflight_vdn_official200 import (
    DEFAULT_OUTPUT as DEFAULT_PREFLIGHT,
    DEFAULT_RUN_ROOT,
    _fresh_content_inventory_identity,
    _strict_json,
    validate_preflight_report,
)
from experiments.train_vdn_official200 import (
    _best_artifact_from_state,
    _summary_from_state,
)
from experiments.vdn_baseline import (
    build_vdn_model,
    grouped_train_val_split,
    load_syncg_manifest,
    set_random_seed,
    sha256_file,
    sha256_source_file,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_EPOCHS,
    OFFICIAL200_FORMAL_SEEDS,
    OFFICIAL200_IMAGE_SIZE,
    OFFICIAL200_PROTOCOL,
    OFFICIAL200_SCHEMA_VERSION,
    OFFICIAL200_STOPPING_POLICY,
    OFFICIAL200_VALIDATION_FRACTION,
    OFFICIAL200_VERIFICATION_PROTOCOL,
    build_official200_signature,
    canonical_json_sha256,
    assert_syncg_train_manifest_path,
    assert_train_only_path,
    model_state_sha256,
    official200_source_hashes,
    require_formal_seed,
    strict_model_state_health,
    validate_authorized_runtime_environment,
    validate_authoritative_checkpoint,
    validate_determinism_report_payload,
)
from experiments.vdn_phase2_protocol import nested_state_equal


VERIFICATION_SCHEMA_KEYS = frozenset(
    {
        "protocol",
        "schema_version",
        "verified",
        "training_artifacts_verified",
        "eligible_for_three_seed_cohort",
        "supporting_test_evaluation_authorized",
        "field_confirmatory_evaluation_authorized",
        "run_dir",
        "seed",
        "epochs",
        "best_epoch",
        "best_validation_angle_mae_degrees",
        "tail_diagnostic",
        "stopping_policy",
        "official_stopping_boundary_reached",
        "additional_training_authorized",
        "phase4_authorized",
        "optimizer_attempts",
        "optimizer_steps",
        "skipped_optimizer_steps",
        "skipped_optimizer_step_rate",
        "full_run_max_skipped_optimizer_steps",
        "preflight",
        "content_inventory",
        "determinism_policy",
        "determinism_authorization",
        "runtime_environment",
        "source_hash_protocol",
        "training_source_sha256",
        "best_checkpoint_sha256",
        "last_checkpoint_sha256",
        "summary_sha256",
        "verifier_source_sha256",
        "authoritative_checkpoint_health",
        "best_model_health",
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
        "canonical_verification_payload_sha256",
    }
)


def write_json_no_clobber(value: Mapping[str, Any], output: Path) -> Path:
    output = assert_train_only_path(
        output,
        label="official-200 verification output",
    )
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite official-200 verification: {output}"
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
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite official-200 verification: {output}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


@contextmanager
def exclusive_verification_lock(run_dir: Path) -> Iterator[None]:
    run_dir = assert_train_only_path(
        run_dir,
        label="official-200 run directory",
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    lock_path = run_dir / "writer.lock"
    payload = json.dumps(
        {
            "attempt": uuid.uuid4().hex,
            "pid": os.getpid(),
            "protocol": "vdn_official200_exclusive_verification_lock_v1",
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
            f"official-200 run already has a writer lock: {lock_path}"
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


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a checkpoint mapping")
    return value


def _verify_locked(
    *,
    run_dir: Path,
    run_root: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path,
    preflight_path: Path,
    determinism_report_path: Path,
) -> dict[str, Any]:
    run_dir = assert_train_only_path(
        run_dir,
        label="official-200 run directory",
    )
    run_root = assert_train_only_path(
        run_root,
        label="official-200 run root",
    )
    manifest = assert_syncg_train_manifest_path(manifest)
    vdn_source = Path(vdn_source).resolve()
    content_inventory = assert_train_only_path(
        content_inventory,
        label="official-200 content inventory",
    )
    preflight_path = assert_train_only_path(
        preflight_path,
        label="official-200 preflight report",
    )
    determinism_report_path = assert_train_only_path(
        determinism_report_path,
        label="official-200 determinism report",
    )
    paths = {
        "summary": run_dir / "summary.json",
        "last": run_dir / "last.pt",
        "best": run_dir / "best.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete official-200 run artifacts: {missing}"
        )
    summary = _strict_json(paths["summary"])
    signature = summary.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("official-200 summary signature is absent")
    seed = require_formal_seed(int(signature.get("seed", -1)))
    if run_dir != run_root / f"seed_{seed}":
        raise ValueError("official-200 run directory/seed identity drifted")

    preflight, preflight_binding = validate_preflight_report(
        preflight_path,
        manifest=manifest,
        vdn_source=vdn_source,
        content_inventory=content_inventory,
        run_root=run_root,
        require_output_absent=False,
    )
    determinism_report = _strict_json(determinism_report_path)
    determinism_authorization = validate_determinism_report_payload(
        determinism_report,
        report_path=determinism_report_path,
        preflight_binding=preflight_binding,
        content_inventory_identity=preflight["content_inventory"],
        vdn_source=vdn_source,
    )
    fresh_content = _fresh_content_inventory_identity(
        report_path=content_inventory,
        manifest=manifest,
        workers=1,
    )
    if fresh_content != preflight["content_inventory"]:
        raise ValueError(
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
    initial_hash = model_state_sha256(model.state_dict())
    if initial_hash != run_plan["initial_model_state_sha256"]:
        raise ValueError(
            "official-200 initial model no longer matches preflight"
        )
    runtime_environment = signature.get("runtime_environment")
    if not isinstance(runtime_environment, Mapping):
        raise ValueError(
            "official-200 signed runtime environment is absent"
        )
    runtime_authorization = validate_authorized_runtime_environment(
        determinism_report,
        runtime_environment,
    )
    if runtime_environment.get("determinism_authorization") != (
        runtime_authorization
    ):
        raise ValueError(
            "official-200 signed runtime authorization drifted"
        )
    expected_signature = build_official200_signature(
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
        initialization_state_sha256=initial_hash,
        content_inventory_identity=fresh_content,
        preflight_binding=preflight_binding,
        determinism_authorization=determinism_authorization,
        runtime_environment=runtime_environment,
        source_sha256=official200_source_hashes(vdn_source),
    )
    if dict(signature) != expected_signature:
        raise ValueError("official-200 training signature drifted")

    last = _load_torch_mapping(paths["last"])
    checkpoint_health = validate_authoritative_checkpoint(
        last,
        model=model,
        train_samples=train_samples,
        signature=expected_signature,
    )
    if int(checkpoint_health["epoch"]) != OFFICIAL200_EPOCHS:
        raise ValueError("official-200 run did not reach epoch 200")
    expected_summary = _summary_from_state(last)
    if summary != expected_summary:
        raise ValueError("official-200 summary is stale or inconsistent")
    best = _load_torch_mapping(paths["best"])
    expected_best = _best_artifact_from_state(last)
    if set(best) != set(expected_best):
        raise ValueError("official-200 best checkpoint schema drifted")
    for field in set(expected_best) - {"model_state"}:
        if best[field] != expected_best[field]:
            raise ValueError(
                f"official-200 best checkpoint {field} drifted"
            )
    if not nested_state_equal(
        best.get("model_state"),
        expected_best["model_state"],
    ):
        raise ValueError("official-200 best model state drifted")
    best_health = strict_model_state_health(
        model,
        best["model_state"],
        label="official-200 derived best",
    )

    history_health = checkpoint_health["history"]
    attempts = int(history_health["cumulative_attempted_optimizer_steps"])
    successful = int(history_health["cumulative_optimizer_steps"])
    skipped = int(history_health["cumulative_skipped_optimizer_steps"])
    if successful + skipped != attempts:
        raise ValueError("official-200 optimizer accounting does not close")
    diagnostic = summary.get("tail_diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("official-200 tail diagnostic is absent")
    if (
        diagnostic.get("authorization_gate") is not False
        or diagnostic.get("additional_training_authorized") is not False
        or diagnostic.get("phase4_authorized") is not False
    ):
        raise ValueError(
            "official-200 tail diagnostic changed the stopping boundary"
        )

    report: dict[str, Any] = {
        "protocol": OFFICIAL200_VERIFICATION_PROTOCOL,
        "schema_version": OFFICIAL200_SCHEMA_VERSION,
        "verified": True,
        "training_artifacts_verified": True,
        "eligible_for_three_seed_cohort": True,
        "supporting_test_evaluation_authorized": False,
        "field_confirmatory_evaluation_authorized": False,
        "run_dir": str(run_dir),
        "seed": seed,
        "epochs": OFFICIAL200_EPOCHS,
        "best_epoch": int(summary["best_epoch"]),
        "best_validation_angle_mae_degrees": float(
            summary["best_validation_angle_mae_degrees"]
        ),
        "tail_diagnostic": diagnostic,
        "stopping_policy": OFFICIAL200_STOPPING_POLICY,
        "official_stopping_boundary_reached": True,
        "additional_training_authorized": False,
        "phase4_authorized": False,
        "optimizer_attempts": attempts,
        "optimizer_steps": successful,
        "skipped_optimizer_steps": skipped,
        "skipped_optimizer_step_rate": skipped / attempts,
        "full_run_max_skipped_optimizer_steps": int(
            history_health["full_run_max_skipped_optimizer_steps"]
        ),
        "preflight": preflight_binding,
        "content_inventory": fresh_content,
        "determinism_policy": expected_signature["determinism"],
        "determinism_authorization": determinism_authorization,
        "runtime_environment": runtime_environment,
        "source_hash_protocol": expected_signature[
            "source_hash_protocol"
        ],
        "training_source_sha256": expected_signature["source_sha256"],
        "best_checkpoint_sha256": sha256_file(paths["best"]),
        "last_checkpoint_sha256": sha256_file(paths["last"]),
        "summary_sha256": sha256_file(paths["summary"]),
        "verifier_source_sha256": sha256_source_file(
            Path(__file__).resolve()
        ),
        "authoritative_checkpoint_health": checkpoint_health,
        "best_model_health": best_health,
        "test_data_opened_or_read": False,
        "public_data_opened_or_read": False,
        "field_data_opened_or_read": False,
        "sealed_data_opened_or_read": False,
        "confirmatory_data_opened_or_read": False,
    }
    report["canonical_verification_payload_sha256"] = (
        canonical_json_sha256(report)
    )
    if set(report) != VERIFICATION_SCHEMA_KEYS:
        raise RuntimeError(
            "official-200 verification schema drifted internally"
        )
    return report


def verify_official200_run(
    run_dir: Path,
    *,
    run_root: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path,
    preflight_path: Path,
    determinism_report_path: Path,
) -> dict[str, Any]:
    with exclusive_verification_lock(run_dir):
        return _verify_locked(
            run_dir=run_dir,
            run_root=run_root,
            manifest=manifest,
            vdn_source=vdn_source,
            content_inventory=content_inventory,
            preflight_path=preflight_path,
            determinism_report_path=determinism_report_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = assert_train_only_path(
        args.output,
        label="official-200 verification output",
    )
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite official-200 verification: {output}"
        )
    report = verify_official200_run(
        args.run_dir,
        run_root=args.run_root,
        manifest=args.manifest,
        vdn_source=args.vdn_source,
        content_inventory=args.content_inventory,
        preflight_path=args.preflight,
        determinism_report_path=args.determinism_report,
    )
    write_json_no_clobber(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
