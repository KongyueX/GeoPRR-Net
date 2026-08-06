"""Read-only gates for the final five-method field-blind event chain.

The public gate independently verifies the completed five-method
materialization, its exact roster path, and the shared public frontend.  The
dataset gate deliberately reads only the owner-frozen identity metadata: it
compares explicit manifest/label paths with their already-declared bindings
without opening, hashing, listing, or otherwise probing either bound file.

The optional runtime gate instantiates the public detector and all five frozen
full-reading adapters, but accepts no dataset path.  It is intended to catch a
missing dependency, bad checkpoint, or GPU-memory failure before an
irreversible blind-inference claim is created.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from experiments.field_blind_multimethod import (
    METHOD_ROLES,
    _load_adapters,
    _load_dataset_identity,
    _load_shared_detector,
    load_frontend_plan,
    load_method_roster,
)
from experiments.materialize_field_blind_bundles import (
    MATERIALIZATION_PROTOCOL,
    verify_materialization,
)
from experiments.v5_unified_full_auto_adapter import sha256_file
from experiments.v5_unified_two_stage_retest import strict_json_load


PROTOCOL: Final[str] = "field_blind_final_chain_preflight_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _absolute_unopened(path: Path, *, label: str) -> Path:
    """Normalize a declared path without testing whether its target exists."""

    value = Path(path)
    _require(value.is_absolute(), f"{label} path must be explicit and absolute")
    return value.resolve(strict=False)


def _verify_exact_materialization_seal(root: Path) -> dict[str, Any]:
    """Require the materialization seal to bind every fixed public artifact."""

    seal_path = (root / "seal.json").resolve(strict=True)
    seal = strict_json_load(seal_path)
    _require(isinstance(seal, Mapping), "materialization seal is not an object")
    _require(
        seal.get("protocol") == MATERIALIZATION_PROTOCOL,
        "materialization seal protocol drift",
    )
    _require(seal.get("status") == "sealed", "materialization seal is not sealed")
    artifacts = seal.get("artifacts")
    _require(isinstance(artifacts, Mapping), "materialization seal artifacts absent")
    expected_paths = {
        "summary": root / "summary.json",
        "method_roster": root / "method_roster.json",
        **{
            f"bundle_{role}": root / f"{role}.bundle.json"
            for role in METHOD_ROLES
        },
    }
    _require(
        set(artifacts) == set(expected_paths),
        "materialization seal must bind exactly the summary, roster, and five bundles",
    )
    bindings: dict[str, dict[str, str]] = {}
    for name, expected in expected_paths.items():
        raw = artifacts[name]
        _require(isinstance(raw, Mapping), f"sealed artifact {name} is invalid")
        bound_path = Path(str(raw.get("path") or "")).resolve(strict=True)
        expected_path = expected.resolve(strict=True)
        digest = str(raw.get("sha256") or "").casefold()
        _require(bound_path == expected_path, f"sealed artifact path drift: {name}")
        _require(
            len(digest) == 64
            and set(digest).issubset(frozenset("0123456789abcdef")),
            f"sealed artifact digest is invalid: {name}",
        )
        _require(sha256_file(bound_path) == digest, f"sealed artifact hash drift: {name}")
        bindings[name] = {"path": str(bound_path), "sha256": digest}
    return {
        "path": str(seal_path),
        "sha256": sha256_file(seal_path),
        "artifacts": bindings,
    }


def validate_public_materialization(
    *,
    materialized_root: Path,
    method_roster_path: Path,
    frontend_plan_path: Path,
) -> dict[str, Any]:
    """Authenticate completed public/model artifacts; never accept data paths."""

    root = Path(materialized_root).resolve(strict=True)
    _require(root.is_dir(), "materialized root is not a directory")
    roster_path = Path(method_roster_path).resolve(strict=True)
    frontend_path = Path(frontend_plan_path).resolve(strict=True)
    expected_roster = (root / "method_roster.json").resolve(strict=True)
    _require(roster_path == expected_roster, "explicit method roster is not the sealed materialized roster")

    verified = verify_materialization(root)
    _require(verified.get("status") == "verified", "materialization status is not verified")
    _require(verified.get("verified") is True, "materialization verification failed")
    _require(tuple(verified.get("method_roles") or ()) == METHOD_ROLES, "five-method role order drift")
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(verified.get(key) is False, f"materialization reports restricted access: {key}")
    exact_seal = _verify_exact_materialization_seal(root)

    raw_roster = strict_json_load(roster_path)
    _require(isinstance(raw_roster, Mapping), "materialized roster is not an object")
    shared_frontend = raw_roster.get("shared_frontend")
    _require(isinstance(shared_frontend, Mapping), "materialized roster lacks shared frontend")
    plan_binding = shared_frontend.get("plan")
    _require(isinstance(plan_binding, Mapping), "shared frontend plan binding is absent")
    bound_path = Path(str(plan_binding.get("path") or "")).resolve(strict=True)
    bound_sha256 = str(plan_binding.get("sha256") or "").casefold()
    _require(frontend_path == bound_path, "explicit frontend is not the materialized shared frontend")
    _require(len(bound_sha256) == 64, "shared frontend digest is invalid")
    _require(sha256_file(frontend_path) == bound_sha256, "shared frontend hash drift")

    loaded_roster_path, _, bundles = load_method_roster(roster_path)
    _require(loaded_roster_path == roster_path, "loaded roster path drift")
    loaded_frontend_path, _ = load_frontend_plan(frontend_path)
    _require(loaded_frontend_path == frontend_path, "loaded frontend path drift")
    _require(tuple(bundles) == METHOD_ROLES, "loaded five-method roster order drift")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "public_materialization_verified",
        "materialized_root": str(root),
        "method_roster": {"path": str(roster_path), "sha256": sha256_file(roster_path)},
        "frontend_plan": {"path": str(frontend_path), "sha256": sha256_file(frontend_path)},
        "materialization_seal": {
            "path": exact_seal["path"],
            "sha256": exact_seal["sha256"],
        },
        "sealed_bundle_count": sum(
            name.startswith("bundle_") for name in exact_seal["artifacts"]
        ),
        "method_roles": list(METHOD_ROLES),
        "garc_shared_range_binding_sha256": verified[
            "garc_shared_range_binding_sha256"
        ],
        "field_manifest_opened": False,
        "field_images_opened": False,
        "field_labels_opened": False,
    }


def validate_explicit_dataset_bindings(
    *,
    dataset_identity_path: Path,
    manifest_path: Path,
    labels_path: Path,
) -> dict[str, Any]:
    """Bind explicit paths to owner metadata without opening either data file."""

    identity_path = Path(dataset_identity_path).resolve(strict=True)
    explicit_manifest = _absolute_unopened(manifest_path, label="manifest")
    explicit_labels = _absolute_unopened(labels_path, label="labels")
    _require(explicit_manifest != explicit_labels, "manifest and labels must be separate")
    dataset = _load_dataset_identity(identity_path)
    declared_manifest = Path(dataset["manifest"]["path"]).resolve(strict=False)
    declared_labels = Path(dataset["labels"]["path"]).resolve(strict=False)
    _require(explicit_manifest == declared_manifest, "explicit manifest path differs from owner-frozen identity")
    _require(explicit_labels == declared_labels, "explicit labels path differs from owner-frozen identity")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "explicit_owner_bindings_verified_without_data_access",
        "dataset_identity": dataset["identity"],
        "declared_images": int(dataset["cohort"]["declared_images"]),
        "manifest": {
            "path": str(explicit_manifest),
            "declared_sha256": dataset["manifest"]["sha256"],
            "opened": False,
            "hashed": False,
        },
        "labels": {
            "path": str(explicit_labels),
            "declared_sha256": dataset["labels"]["sha256"],
            "opened": False,
            "hashed": False,
        },
        "authorization": dict(dataset["authorization"]),
        "field_manifest_opened": False,
        "field_images_opened": False,
        "field_labels_opened": False,
    }


def runtime_preflight(
    *, method_roster_path: Path, frontend_plan_path: Path
) -> dict[str, Any]:
    """Instantiate every frozen runtime without accepting or opening a dataset."""

    roster_path, roster, bundles = load_method_roster(method_roster_path)
    frontend_path, frontend = load_frontend_plan(frontend_plan_path)
    detector = _load_shared_detector(frontend)
    adapters = _load_adapters(roster, bundles)
    _require(tuple(adapters) == METHOD_ROLES, "not all five adapters instantiated")
    # Keep references alive until every identity has been authenticated.
    identities = {
        role: dict(adapters[role].identity) for role in METHOD_ROLES
    }
    _require(detector is not None, "shared detector instantiation returned null")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "all_public_runtimes_instantiated_without_dataset_access",
        "method_roster": {"path": str(roster_path), "sha256": sha256_file(roster_path)},
        "frontend_plan": {"path": str(frontend_path), "sha256": sha256_file(frontend_path)},
        "method_identities": identities,
        "method_roles": list(METHOD_ROLES),
        "shared_detector_instantiated": True,
        "field_manifest_opened": False,
        "field_images_opened": False,
        "field_labels_opened": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    public = commands.add_parser("public-materialization")
    public.add_argument("--materialized-root", type=Path, required=True)
    public.add_argument("--method-roster", type=Path, required=True)
    public.add_argument("--frontend-plan", type=Path, required=True)
    dataset = commands.add_parser("dataset-bindings")
    dataset.add_argument("--dataset-identity", type=Path, required=True)
    dataset.add_argument("--manifest", type=Path, required=True)
    dataset.add_argument("--labels", type=Path, required=True)
    runtime = commands.add_parser("runtime")
    runtime.add_argument("--method-roster", type=Path, required=True)
    runtime.add_argument("--frontend-plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "public-materialization":
        result = validate_public_materialization(
            materialized_root=args.materialized_root,
            method_roster_path=args.method_roster,
            frontend_plan_path=args.frontend_plan,
        )
    elif args.command == "dataset-bindings":
        result = validate_explicit_dataset_bindings(
            dataset_identity_path=args.dataset_identity,
            manifest_path=args.manifest,
            labels_path=args.labels,
        )
    else:
        result = runtime_preflight(
            method_roster_path=args.method_roster,
            frontend_plan_path=args.frontend_plan,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "PROTOCOL",
    "runtime_preflight",
    "validate_explicit_dataset_bindings",
    "validate_public_materialization",
]
