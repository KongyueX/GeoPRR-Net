"""Freeze the train-only preflight for three from-scratch VDN 200-epoch runs."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.freeze_vdn_phase2_syncg_train_inventory import (
    DEFAULT_OUTPUT as DEFAULT_CONTENT_INVENTORY,
    SYNCG_TRAIN_EXPECTED_ROWS,
    verify_inventory_artifact,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    build_vdn_model,
    grouped_train_val_split,
    load_official_resnet18_initialization,
    load_syncg_manifest,
    sample_ids_hash,
    set_random_seed,
    sha256_file,
    verify_vdn_source,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_BATCH_SIZE,
    OFFICIAL200_CONTENT_INVENTORY_ROWS,
    OFFICIAL200_DETERMINISM_POLICY,
    OFFICIAL200_EPOCHS,
    OFFICIAL200_FORMAL_SEEDS,
    OFFICIAL200_IMAGE_SIZE,
    OFFICIAL200_LR_SEGMENTS,
    OFFICIAL200_MILESTONES,
    OFFICIAL200_PREFLIGHT_PROTOCOL,
    OFFICIAL200_PROTOCOL,
    OFFICIAL200_ROTATION_FACTOR,
    OFFICIAL200_SCALE_FACTOR,
    OFFICIAL200_SCHEMA_VERSION,
    OFFICIAL200_SCOPE,
    OFFICIAL200_STOPPING_POLICY,
    OFFICIAL200_VALIDATION_FRACTION,
    OFFICIAL200_WEIGHT_DECAY,
    OFFICIAL200_WORKERS,
    apply_phase2_determinism_policy,
    assert_syncg_train_manifest_path,
    assert_train_only_path,
    canonical_json_sha256,
    model_state_sha256,
    normalize_content_inventory_identity,
    official200_source_hashes,
)


DEFAULT_RUN_ROOT = (
    PROJECT_DIR / "artifacts" / "runs" / "vdn_syncg_official200"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "protocols"
    / "vdn_official200_preflight_v1.json"
)
PREFLIGHT_SCHEMA_KEYS = frozenset(
    {
        "protocol",
        "schema_version",
        "status",
        "training_authorized",
        "scope",
        "report_path",
        "run_root",
        "formal_seeds",
        "config",
        "manifest",
        "content_inventory",
        "vdn_source",
        "initialization",
        "runs",
        "determinism_policy",
        "source_hash_protocol",
        "source_sha256",
        "output_preconditions",
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
        "cuda_tensor_operations_executed",
        "canonical_preflight_payload_sha256",
    }
)


def _fresh_content_inventory_identity(
    *,
    report_path: Path,
    manifest: Path,
    workers: int,
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
        workers=int(workers),
    )
    return normalize_content_inventory_identity(
        verification,
        inventory_tool_source=inventory_tool,
        manifest=manifest,
    )


def _strict_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON value {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: preflight root is not an object")

    def require_finite(child: Any, location: str) -> None:
        if isinstance(child, float):
            if not math.isfinite(child):
                raise ValueError(
                    f"{path}: non-finite numeric value at {location}"
                )
        elif isinstance(child, Mapping):
            for key, grandchild in child.items():
                require_finite(grandchild, f"{location}.{key}")
        elif isinstance(child, list):
            for index, grandchild in enumerate(child):
                require_finite(grandchild, f"{location}[{index}]")

    require_finite(value, "$")
    return value


def write_json_no_clobber(value: Mapping[str, Any], output: Path) -> Path:
    output = assert_train_only_path(
        output,
        label="official-200 preflight output",
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")
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
            f"refusing to overwrite preflight: {output}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _config_identity() -> dict[str, Any]:
    return {
        "training_protocol": OFFICIAL200_PROTOCOL,
        "epochs": OFFICIAL200_EPOCHS,
        "milestones": list(OFFICIAL200_MILESTONES),
        "learning_rate_segments": [
            dict(segment) for segment in OFFICIAL200_LR_SEGMENTS
        ],
        "batch_size": OFFICIAL200_BATCH_SIZE,
        "workers": OFFICIAL200_WORKERS,
        "image_size": OFFICIAL200_IMAGE_SIZE,
        "validation_fraction": OFFICIAL200_VALIDATION_FRACTION,
        "scale_factor": OFFICIAL200_SCALE_FACTOR,
        "rotation_factor": OFFICIAL200_ROTATION_FACTOR,
        "weight_decay": OFFICIAL200_WEIGHT_DECAY,
        "mixed_precision": True,
        "imagenet_pretrained": True,
        "determinism_authorization": {
            "protocol": (
                "vdn_official200_full_epoch_determinism_probe_v1"
            ),
            "replicates": 2,
            "complete_train_and_validation_epoch": 1,
            "separate_python_processes": True,
            "exact_semantic_equality_required": True,
            "required_before_training": True,
        },
        "stopping_policy": OFFICIAL200_STOPPING_POLICY,
    }


def _require_output_dirs_absent(run_root: Path) -> list[dict[str, Any]]:
    run_root = assert_train_only_path(
        run_root,
        label="official-200 run root",
    )
    result = []
    for seed in OFFICIAL200_FORMAL_SEEDS:
        run_dir = run_root / f"seed_{seed}"
        if run_dir.exists():
            raise FileExistsError(
                f"official-200 run directory already exists: {run_dir}"
            )
        result.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "absent_at_preflight": True,
            }
        )
    return result


def build_preflight_report(
    *,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path,
    run_root: Path,
    output: Path,
    inventory_workers: int,
) -> dict[str, Any]:
    """Build the frozen report without executing a CUDA tensor operation."""

    manifest = assert_syncg_train_manifest_path(manifest)
    vdn_source = Path(vdn_source).resolve()
    content_inventory = assert_train_only_path(
        content_inventory,
        label="official-200 content inventory",
    )
    run_root = assert_train_only_path(
        run_root,
        label="official-200 run root",
    )
    output = assert_train_only_path(
        output,
        label="official-200 preflight output",
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")
    if os.environ.get("PYTHONHASHSEED") != str(
        OFFICIAL200_FORMAL_SEEDS[0]
    ):
        raise RuntimeError(
            f"PYTHONHASHSEED must be {OFFICIAL200_FORMAL_SEEDS[0]} "
            "before official-200 preflight"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before preflight"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before official-200 preflight"
        )
    determinism = apply_phase2_determinism_policy()
    if determinism != OFFICIAL200_DETERMINISM_POLICY:
        raise RuntimeError("official-200 deterministic policy drifted")

    output_preconditions = _require_output_dirs_absent(run_root)
    vdn_commit = verify_vdn_source(vdn_source)
    content_identity = _fresh_content_inventory_identity(
        report_path=content_inventory,
        manifest=manifest,
        workers=inventory_workers,
    )
    samples, manifest_protocol = load_syncg_manifest(
        manifest,
        expected_split="train",
    )
    if len(samples) != OFFICIAL200_CONTENT_INVENTORY_ROWS:
        raise ValueError(
            "official-200 manifest must contain exactly 16000 train rows"
        )
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    _, initialization_checkpoint = load_official_resnet18_initialization()
    initialization_checkpoint = initialization_checkpoint.resolve()

    runs = []
    for seed in OFFICIAL200_FORMAL_SEEDS:
        train_samples, validation_samples = grouped_train_val_split(
            samples,
            validation_fraction=OFFICIAL200_VALIDATION_FRACTION,
            seed=seed,
        )
        hashes = []
        for _ in range(2):
            set_random_seed(seed)
            model = build_vdn_model(
                vdn_source,
                image_size=OFFICIAL200_IMAGE_SIZE,
                imagenet_pretrained=True,
            )
            hashes.append(model_state_sha256(model.state_dict()))
            del model
        if hashes[0] != hashes[1]:
            raise RuntimeError(
                f"official-200 seed {seed} initialization is not reproducible"
            )
        runs.append(
            {
                "seed": seed,
                "run_dir": str(run_root / f"seed_{seed}"),
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "train_sample_ids_sha256": sample_ids_hash(train_samples),
                "validation_sample_ids_sha256": sample_ids_hash(
                    validation_samples
                ),
                "initial_model_state_sha256": hashes[0],
                "initialization_rebuild_exact": True,
            }
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "official-200 preflight unexpectedly initialized CUDA"
        )

    report: dict[str, Any] = {
        "protocol": OFFICIAL200_PREFLIGHT_PROTOCOL,
        "schema_version": OFFICIAL200_SCHEMA_VERSION,
        "status": "passed",
        "training_authorized": True,
        "scope": OFFICIAL200_SCOPE,
        "report_path": str(output),
        "run_root": str(run_root),
        "formal_seeds": list(OFFICIAL200_FORMAL_SEEDS),
        "config": _config_identity(),
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "dataset": manifest_protocol.get("dataset"),
            "split": manifest_protocol.get("split"),
            "rows": len(samples),
        },
        "content_inventory": content_identity,
        "vdn_source": {
            "path": str(vdn_source),
            "commit": vdn_commit,
        },
        "initialization": {
            "checkpoint_path": str(initialization_checkpoint),
            "checkpoint_sha256": sha256_file(initialization_checkpoint),
            "per_seed_rebuilds": 2,
            "all_rebuilds_exact": True,
        },
        "runs": runs,
        "determinism_policy": determinism,
        "source_hash_protocol": "utf8_source_newlines_lf_v1",
        "source_sha256": official200_source_hashes(vdn_source),
        "output_preconditions": output_preconditions,
        "test_data_opened_or_read": False,
        "public_data_opened_or_read": False,
        "field_data_opened_or_read": False,
        "sealed_data_opened_or_read": False,
        "confirmatory_data_opened_or_read": False,
        "cuda_tensor_operations_executed": False,
    }
    report["canonical_preflight_payload_sha256"] = canonical_json_sha256(
        report
    )
    if set(report) != PREFLIGHT_SCHEMA_KEYS:
        raise RuntimeError("official-200 preflight schema drifted internally")
    return report


def validate_preflight_report(
    report_path: Path,
    *,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path,
    run_root: Path,
    require_output_absent: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a frozen report and return it plus its signature binding."""

    report_path = assert_train_only_path(
        report_path,
        label="official-200 preflight report",
    )
    manifest = assert_syncg_train_manifest_path(manifest)
    vdn_source = Path(vdn_source).resolve()
    content_inventory = assert_train_only_path(
        content_inventory,
        label="official-200 content inventory",
    )
    run_root = assert_train_only_path(
        run_root,
        label="official-200 run root",
    )
    report = _strict_json(report_path)
    if set(report) != PREFLIGHT_SCHEMA_KEYS:
        raise ValueError("official-200 preflight schema drifted")
    if (
        report.get("protocol") != OFFICIAL200_PREFLIGHT_PROTOCOL
        or int(report.get("schema_version", -1))
        != OFFICIAL200_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("training_authorized") is not True
        or report.get("scope") != OFFICIAL200_SCOPE
        or report.get("report_path") != str(report_path)
        or report.get("run_root") != str(run_root)
        or report.get("formal_seeds") != list(OFFICIAL200_FORMAL_SEEDS)
        or report.get("config") != _config_identity()
        or report.get("determinism_policy")
        != OFFICIAL200_DETERMINISM_POLICY
        or report.get("source_hash_protocol")
        != "utf8_source_newlines_lf_v1"
    ):
        raise ValueError("official-200 preflight identity drifted")
    for field in (
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
        "cuda_tensor_operations_executed",
    ):
        if report.get(field) is not False:
            raise ValueError(
                f"official-200 preflight provenance flag drifted: {field}"
            )
    canonical = report.get("canonical_preflight_payload_sha256")
    payload = dict(report)
    payload.pop("canonical_preflight_payload_sha256", None)
    if canonical != canonical_json_sha256(payload):
        raise ValueError("official-200 preflight canonical digest drifted")
    if report.get("source_sha256") != official200_source_hashes(vdn_source):
        raise ValueError("official-200 source changed after preflight")
    manifest_record = report.get("manifest")
    if not isinstance(manifest_record, Mapping) or manifest_record != {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "protocol_path": str(
            manifest.with_name(manifest.name + ".protocol.json")
        ),
        "protocol_sha256": sha256_file(
            manifest.with_name(manifest.name + ".protocol.json")
        ),
        "dataset": "SyncG",
        "split": "train",
        "rows": OFFICIAL200_CONTENT_INVENTORY_ROWS,
    }:
        raise ValueError("official-200 preflight manifest identity drifted")
    if verify_vdn_source(vdn_source) != (
        report.get("vdn_source") or {}
    ).get("commit"):
        raise ValueError("official-200 VDN source identity drifted")
    if (report.get("vdn_source") or {}).get("path") != str(vdn_source):
        raise ValueError("official-200 VDN source path drifted")
    content = report.get("content_inventory")
    try:
        relative_content_path = content_inventory.relative_to(PROJECT_DIR)
    except ValueError as exc:
        raise ValueError(
            "official-200 content inventory must remain inside the project"
        ) from exc
    expected_content_report_paths = {
        str(relative_content_path),
        relative_content_path.as_posix(),
    }
    if (
        not isinstance(content, Mapping)
        or content.get("report_path") not in expected_content_report_paths
        or content.get("report_sha256") != sha256_file(content_inventory)
        or content.get("fresh_content_rehashed") is not True
        or int(content.get("rows", -1))
        != OFFICIAL200_CONTENT_INVENTORY_ROWS
    ):
        raise ValueError("official-200 content inventory binding drifted")
    initialization = report.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("official-200 initialization record is absent")
    checkpoint = Path(str(initialization.get("checkpoint_path", "")))
    if (
        not checkpoint.is_file()
        or initialization.get("checkpoint_sha256") != sha256_file(checkpoint)
        or initialization.get("all_rebuilds_exact") is not True
        or int(initialization.get("per_seed_rebuilds", -1)) != 2
    ):
        raise ValueError("official-200 initialization identity drifted")

    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    expected_runs = []
    for seed in OFFICIAL200_FORMAL_SEEDS:
        train_samples, validation_samples = grouped_train_val_split(
            samples,
            validation_fraction=OFFICIAL200_VALIDATION_FRACTION,
            seed=seed,
        )
        expected_runs.append(
            {
                "seed": seed,
                "run_dir": str(run_root / f"seed_{seed}"),
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "train_sample_ids_sha256": sample_ids_hash(train_samples),
                "validation_sample_ids_sha256": sample_ids_hash(
                    validation_samples
                ),
                "initial_model_state_sha256": next(
                    str(run["initial_model_state_sha256"])
                    for run in report["runs"]
                    if int(run["seed"]) == seed
                ),
                "initialization_rebuild_exact": True,
            }
        )
    if report.get("runs") != expected_runs:
        raise ValueError("official-200 preflight run identity drifted")
    expected_preconditions = [
        {
            "seed": seed,
            "run_dir": str(run_root / f"seed_{seed}"),
            "absent_at_preflight": True,
        }
        for seed in OFFICIAL200_FORMAL_SEEDS
    ]
    if report.get("output_preconditions") != expected_preconditions:
        raise ValueError("official-200 output preconditions drifted")
    if require_output_absent:
        _require_output_dirs_absent(run_root)

    binding = {
        "protocol": OFFICIAL200_PREFLIGHT_PROTOCOL,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "canonical_payload_sha256": canonical,
    }
    return report, binding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=DEFAULT_CONTENT_INVENTORY,
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inventory-workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inventory_workers <= 0:
        raise ValueError("--inventory-workers must be positive")
    report = build_preflight_report(
        manifest=args.manifest,
        vdn_source=args.vdn_source,
        content_inventory=args.content_inventory,
        run_root=args.run_root,
        output=args.output,
        inventory_workers=args.inventory_workers,
    )
    output = write_json_no_clobber(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
