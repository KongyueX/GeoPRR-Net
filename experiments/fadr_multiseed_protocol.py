"""Frozen identities and fail-closed helpers for the FADR v2 evidence chain.

This module is intentionally independent of model code.  It defines the only
three FADR seeds authorized by the preregistered protocol and the normalized
authorization schema that an upstream PEPD/OOF adapter must emit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.strict_json import strict_json_load, strict_jsonl_load


FADR_SEEDS = (20260722, 20260723, 20260724)
FADR_PRIMARY_SEED = 20260722
PEPD_DIRECTION_SEEDS = (20260720, 20260721, 20260722)

FADR_INPUT_AUTHORIZATION_PROTOCOL = "fadr_multiseed_input_authorization_v2"
FADR_INPUT_PREFLIGHT_PROTOCOL = "fadr_multiseed_input_preflight_v2"
FADR_JOINT_TRAINING_PROTOCOL = "fadr_joint_outer_group_stacking_v2"
FADR_JOINT_LINEAGE_PROTOCOL = "fadr_joint_outer_group_lineage_v2"
FADR_JOINT_VERIFICATION_PROTOCOL = "fadr_joint_outer_group_verification_v2"
FADR_MULTI_SEED_COHORT_PROTOCOL = (
    "fadr_joint_outer_group_multiseed_train_only_cohort_v2"
)
FADR_UDSF_HANDOFF_PROTOCOL = "fadr_joint_outer_group_to_udsf_handoff_v1"

# These are duplicated as frozen interface values instead of importing the
# GPU-oriented PEPD modules.  The preflight compares the values against the
# actual cohort/handoff documents, so a protocol drift fails closed.
EXPECTED_PEPD_COHORT_PROTOCOL = (
    "pepd_syncg_grouped_val_mixed_authoritative_cohort_v2"
)
EXPECTED_PEPD_HANDOFF_PROTOCOL = "pepd_mixed_authoritative_oof_handoff_v2"
EXPECTED_PEPD_COLLECTOR_CONTRACT_PROTOCOL = (
    "pepd_mixed_authoritative_uncertainty_fusion_oof_collector_contract_v2"
)
EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED = {
    "20260720": "pepd_syncg_grouped_val_continuation_v1",
    "20260721": "pepd_syncg_grouped_val_bounded_extension_v2",
    "20260722": "pepd_syncg_grouped_val_continuation_v1",
}
EXPECTED_PEPD_VERIFICATION_PROTOCOL_BY_SEED = {
    "20260720": "pepd_syncg_grouped_val_run_verification_v1",
    "20260721": "pepd_syncg_grouped_val_bounded_extension_verification_v2",
    "20260722": "pepd_syncg_grouped_val_run_verification_v1",
}
EXPECTED_OOF_PROTOCOL = "syncg_probabilistic_uncertainty_fusion_oof_v2"

AUTHORIZATION_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "scope",
        "dataset",
        "split",
        "train_only_certified",
        "group_leakage_count",
        "test_samples_used",
        "public_samples_used",
        "field_samples_used",
        "public_test_field_evaluation_authorized",
        "direction_seeds",
        "fadr_seeds",
        "direction_checkpoint_sha256",
        "direction_run_protocol",
        "direction_verification_protocol",
        "inputs",
        "source_identity",
    }
)
AUTHORIZATION_INPUT_KEYS = frozenset(
    {
        "oof_pairs",
        "oof_metadata",
        "oof_summary",
        "pepd_cohort",
        "pepd_oof_handoff",
    }
)
FILE_IDENTITY_KEYS = frozenset({"path", "sha256"})
AUTHORIZATION_SOURCE_IDENTITY_KEYS = frozenset(
    {"builder", "protocol", "strict_json"}
)

_FORBIDDEN_EXACT_PATH_PARTS = frozenset(
    {
        "public",
        "test",
        "tests",
        "field",
        "field-development",
        "field-confirmatory",
        "field_development",
        "field_confirmatory",
        "sealed",
        "confirmatory",
        "holdout",
    }
)
_FORBIDDEN_PATH_PREFIXES = (
    "public_",
    "public-",
    "test_",
    "test-",
    "field_",
    "field-",
    "sealed_",
    "sealed-",
    "confirmatory_",
    "confirmatory-",
    "holdout_",
    "holdout-",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{label} schema mismatch: missing={missing}, extra={extra}")


def require_exact_int(value: Any, *, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be integer {expected}")


def require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def resolve_declared_path(value: Any, *, project_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def assert_train_only_path(path: Path, *, label: str) -> None:
    """Reject obvious public/test/field/sealed path namespaces."""

    for part in path.parts:
        normalized = part.casefold()
        if normalized in _FORBIDDEN_EXACT_PATH_PARTS or normalized.startswith(
            _FORBIDDEN_PATH_PREFIXES
        ):
            raise ValueError(f"{label} enters forbidden path namespace: {part!r}")


def validate_file_identity(
    record: Any,
    *,
    supplied_path: Path,
    project_root: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a file-identity object")
    require_exact_keys(record, FILE_IDENTITY_KEYS, label=label)
    declared = resolve_declared_path(
        record.get("path"),
        project_root=project_root,
        label=label,
    )
    supplied = supplied_path.resolve()
    if declared != supplied:
        raise ValueError(
            f"{label} path mismatch: authorization={declared}, supplied={supplied}"
        )
    assert_train_only_path(supplied, label=label)
    expected_hash = require_sha256(record.get("sha256"), label=f"{label}.sha256")
    actual_hash = sha256_file(supplied)
    if expected_hash != actual_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    return {"path": str(supplied), "sha256": actual_hash}


def validate_authorization_shape(value: Mapping[str, Any]) -> None:
    require_exact_keys(
        value,
        AUTHORIZATION_TOP_LEVEL_KEYS,
        label="FADR input authorization",
    )
    require_exact_int(
        value.get("schema_version"),
        expected=1,
        label="authorization.schema_version",
    )
    if value.get("protocol") != FADR_INPUT_AUTHORIZATION_PROTOCOL:
        raise ValueError("wrong FADR input authorization protocol")
    if value.get("status") != "authorized":
        raise ValueError("FADR input authorization is not authorized")
    if value.get("scope") != "SyncG/train strict grouped OOF -> FADR train-only":
        raise ValueError("FADR input authorization scope drifted")
    if value.get("dataset") != "SyncG" or value.get("split") != "train":
        raise ValueError("FADR authorization is not restricted to SyncG/train")
    if value.get("train_only_certified") is not True:
        raise ValueError("FADR authorization lacks train-only certification")
    for name in (
        "group_leakage_count",
        "test_samples_used",
        "public_samples_used",
        "field_samples_used",
    ):
        require_exact_int(value.get(name), expected=0, label=f"authorization.{name}")
    if value.get("public_test_field_evaluation_authorized") is not False:
        raise ValueError("FADR authorization must forbid public/test/field evaluation")
    if value.get("direction_seeds") != list(PEPD_DIRECTION_SEEDS):
        raise ValueError("authorization direction seeds drifted")
    if value.get("fadr_seeds") != list(FADR_SEEDS):
        raise ValueError("authorization FADR seeds drifted")

    checkpoints = value.get("direction_checkpoint_sha256")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {
        str(seed) for seed in PEPD_DIRECTION_SEEDS
    }:
        raise ValueError("authorization checkpoint seed set drifted")
    for seed in PEPD_DIRECTION_SEEDS:
        require_sha256(
            checkpoints[str(seed)],
            label=f"authorization.direction_checkpoint_sha256.{seed}",
        )
    if value.get("direction_run_protocol") != EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED:
        raise ValueError("authorization direction run protocols drifted")
    if (
        value.get("direction_verification_protocol")
        != EXPECTED_PEPD_VERIFICATION_PROTOCOL_BY_SEED
    ):
        raise ValueError("authorization direction verification protocols drifted")

    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("authorization.inputs must be an object")
    require_exact_keys(inputs, AUTHORIZATION_INPUT_KEYS, label="authorization.inputs")
    source_identity = value.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("authorization.source_identity must be an object")
    require_exact_keys(
        source_identity,
        AUTHORIZATION_SOURCE_IDENTITY_KEYS,
        label="authorization.source_identity",
    )
    for name in AUTHORIZATION_SOURCE_IDENTITY_KEYS:
        require_sha256(
            source_identity.get(name),
            label=f"authorization.source_identity.{name}",
        )
