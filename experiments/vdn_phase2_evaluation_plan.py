"""Exact allowlist for cohort-authorized VDN Phase-2 public evaluation.

The plan contains no field/sealed path.  Callers must validate the three-seed
cohort before invoking :func:`validate_phase2_evaluation_plan`; only then may
this module hash or open a listed public/test artifact.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from experiments.formal_environment import formal_environment_identity
from experiments.vdn_baseline import PROJECT_DIR, sha256_file, sha256_source_file


PHASE2_EVALUATION_PLAN_PROTOCOL = (
    "vdn_phase2_supporting_public_evaluation_plan_v1"
)
PHASE2_EVALUATION_PLAN_SOURCE_SHA256_PROTOCOL = (
    "utf8_source_newlines_lf_v1"
)
PHASE2_EVALUATION_EXECUTION_SOURCE_PROTOCOL = (
    "vdn_phase2_evaluation_execution_sources_v1"
)
FORMAL_RUN_ROOT = PROJECT_DIR / "artifacts" / "runs" / "vdn_syncg_phase2"
FORMAL_VDN_SOURCE = (
    PROJECT_DIR / "artifacts" / "vendor" / "VectorDetectionNetwork"
)
FORMAL_METER_WEIGHTS = (
    PROJECT_DIR
    / "utils"
    / "angleDetect"
    / "yoloDetection"
    / "result"
    / "yolo_findMeter.pt"
)
FORMAL_POINT_WEIGHTS = (
    PROJECT_DIR
    / "utils"
    / "angleDetect"
    / "yoloDetection"
    / "result"
    / "yolo_pointbest.pt"
)
FORMAL_METER_WEIGHTS_SHA256 = (
    "98b8f40cba170b40f828bd0579e6a8a5feba500fd83261a13651ab428c0956ab"
)
FORMAL_POINT_WEIGHTS_SHA256 = (
    "2cb5c2523e364063ccdfd5c047390f17986ebcb622cef09d8604dfdaf038bdd6"
)
FORMAL_DEGRADATION_SEED = 20260720
FORMAL_BOOTSTRAP_ITERATIONS = 2000
FORMAL_BOOTSTRAP_SEED = 20260724
FORMAL_BATCH_SIZE = 16
FORMAL_PARENT_SEEDS = (20260720, 20260721, 20260722)
PHASE2_PUBLIC_PREFLIGHT_PROTOCOL = (
    "vdn_phase2_public_evidence_preflight_v1"
)
PHASE2_PUBLIC_PREFLIGHT_SCHEMA_VERSION = 2
FORMAL_PUBLIC_PREFLIGHT = (
    PROJECT_DIR
    / "artifacts"
    / "protocols"
    / "vdn_phase2_public_evidence_preflight_v1.json"
)
PINNED_PUBLIC_RELEASE_INVENTORY_SHA256 = {
    "syncg_test": (
        "9ade19323c965d0b5559ce2497f610b9ae4e97516787d35757e08307027a5234"
    ),
    "rpm10k_support": (
        "34abc88269b84fa19f1f11dde561fa17397e6eb9e167bddabb82d731cb99df3e"
    ),
}
PINNED_FINAL_EVALUATION_ARTIFACT_SHA256 = {
    "clean": {
        "predictions": (
            "1bd17a1e8ed4a9ae83cb86e1400ae98e484982d27cca1fe1469c77ebddd8297f"
        ),
        "summary": (
            "cb450d5d337cc33e3548acfecc9b35512f7352c4220a84d5c5cda99c6fe68c35"
        ),
    },
    "blur_moderate": {
        "predictions": (
            "34dc0a7eada2eb662d0eea7ca60e96ad0568f7dc37b096402365c2a473d0b72a"
        ),
        "summary": (
            "f0662fd20a28d805a83cb931bf60d70e922fc90d88b90c3f2001b3482a0bbdfa"
        ),
    },
    "blur_severe": {
        "predictions": (
            "6d3cdbaed788380f6f631aa7c92a9ba736e093a67d997b6709ecd5c339a6de28"
        ),
        "summary": (
            "e2bbc0600337d815cba040de76f11e8c2bc211d88cc50888758f6c0c1003445c"
        ),
    },
    "perspective_moderate": {
        "predictions": (
            "eeb942f290efcb9b86fe59ca89b9464086f93b578789b4d3493548fa7d18fbe5"
        ),
        "summary": (
            "76a631abcb24a7c064f3bab6113c574c65688cb64684a3419d9f11f642235a85"
        ),
    },
    "perspective_severe": {
        "predictions": (
            "ff7d1a91cf0310a7dbb4786d9e30be5b2bcf07e5daeb7829edcf45e9c8fa0431"
        ),
        "summary": (
            "9825e2f9db8a3080febd093c355cd7a0f7efb530635fd1fa9fd7fa917b6d5234"
        ),
    },
    "combined_severe": {
        "predictions": (
            "7e710d632d539d43286b441cbfac01de09aecebd146bbe862ca46a9e3a1147ac"
        ),
        "summary": (
            "238ae12848921351e4cd1b894960a346d9eb05e6ce981e384d739313c5c9085f"
        ),
    },
    "rpm10k": {
        "predictions": (
            "50299f619036db1134b474581404f1d34738635b77b891e4e8a735a3876fd9de"
        ),
        "summary": (
            "b9bd83e91e0adb0a89e7f6efea2f178d0bf733f5b41fa0e0cc8921f17538434b"
        ),
    },
}


def _project_path(relative: str) -> Path:
    return (PROJECT_DIR / Path(relative)).resolve()


_SYNCG_MANIFEST = "artifacts/manifests/syncg_test.jsonl"
_SYNCG_MANIFEST_SHA256 = (
    "e09f099e047a007f44e53ffb89e9103fd25532922f1d24ed2176931f07686f1d"
)
_SYNCG_PROTOCOL_SHA256 = (
    "165beb2e326daa5652f0b604448280e391d26ed483fc1b3307c5cb6ec2cbb1ea"
)
_RPM_MANIFEST = "artifacts/manifests/rpm10k_single_pointer_test.jsonl"
_RPM_MANIFEST_SHA256 = (
    "028d296e2cf12527dfe53aaec01e8b265ffe103e5e6ef159fb04432ab82f8b78"
)
_RPM_PROTOCOL_SHA256 = (
    "29f169739223826dcf54853b94ac4f975959b24692a572371fc23687cd168a0f"
)

PHASE2_EVALUATION_PLANS: Mapping[str, Mapping[str, Any]] = {
    "clean": {
        "manifest": _SYNCG_MANIFEST,
        "manifest_sha256": _SYNCG_MANIFEST_SHA256,
        "manifest_protocol_sha256": _SYNCG_PROTOCOL_SHA256,
        "shared_predictions": (
            "artifacts/predictions/robustness/syncg_test_clean.jsonl"
        ),
        "shared_predictions_sha256": (
            "dd3afb3dffdf1f8dc4a1fb6f39465e2719ff62920bb020282507e31d3d7519f3"
        ),
        "shared_predictions_metadata_sha256": (
            "14abadb53a1c8ee498cce5c60765f6bd2967faf8348956c09a301faba24b8889"
        ),
        "condition": "clean",
    },
    "blur_moderate": {
        "manifest": _SYNCG_MANIFEST,
        "manifest_sha256": _SYNCG_MANIFEST_SHA256,
        "manifest_protocol_sha256": _SYNCG_PROTOCOL_SHA256,
        "shared_predictions": (
            "artifacts/predictions/robustness/"
            "syncg_test_blur_moderate.jsonl"
        ),
        "shared_predictions_sha256": (
            "92d72dc7854fb76bd2abb12cd9866b56a04de96cd222ddae76eba075527c1511"
        ),
        "shared_predictions_metadata_sha256": (
            "3fab369291ac85b536f1edb3b063814a9064613647f0445d4bed9aa60ebf403b"
        ),
        "condition": "blur_moderate",
    },
    "blur_severe": {
        "manifest": _SYNCG_MANIFEST,
        "manifest_sha256": _SYNCG_MANIFEST_SHA256,
        "manifest_protocol_sha256": _SYNCG_PROTOCOL_SHA256,
        "shared_predictions": (
            "artifacts/predictions/robustness/syncg_test_blur_severe.jsonl"
        ),
        "shared_predictions_sha256": (
            "a28cf255657292cbbf7cbe48da545219eb18058f2e8b78e4a77bb9d2b9bf6838"
        ),
        "shared_predictions_metadata_sha256": (
            "2f8ca89814cc8c11f99a1648b4174db1766ea923d03f709513721a44c2a8150e"
        ),
        "condition": "blur_severe",
    },
    "perspective_moderate": {
        "manifest": _SYNCG_MANIFEST,
        "manifest_sha256": _SYNCG_MANIFEST_SHA256,
        "manifest_protocol_sha256": _SYNCG_PROTOCOL_SHA256,
        "shared_predictions": (
            "artifacts/predictions/robustness/"
            "syncg_test_perspective_moderate.jsonl"
        ),
        "shared_predictions_sha256": (
            "b9d921ff7922d3f2697fc1a49f7e3826719a697aa64cdd4e58bf0158f0acfe3e"
        ),
        "shared_predictions_metadata_sha256": (
            "7a742e499d992fea3017779a3a7f07a12e1dd8a0d34dd1a46eebe504373adb1f"
        ),
        "condition": "perspective_moderate",
    },
    "perspective_severe": {
        "manifest": _SYNCG_MANIFEST,
        "manifest_sha256": _SYNCG_MANIFEST_SHA256,
        "manifest_protocol_sha256": _SYNCG_PROTOCOL_SHA256,
        "shared_predictions": (
            "artifacts/predictions/robustness/"
            "syncg_test_perspective_severe.jsonl"
        ),
        "shared_predictions_sha256": (
            "e6b75ca2198f668459ee98e08d5694a7908ecc26599f739363cd1f8bc2c1e5c5"
        ),
        "shared_predictions_metadata_sha256": (
            "53bcb2bafd379b4be85a5242ba7b856e7248d1815e6331b807e03310ccf929d0"
        ),
        "condition": "perspective_severe",
    },
    "combined_severe": {
        "manifest": _SYNCG_MANIFEST,
        "manifest_sha256": _SYNCG_MANIFEST_SHA256,
        "manifest_protocol_sha256": _SYNCG_PROTOCOL_SHA256,
        "shared_predictions": (
            "artifacts/predictions/robustness/"
            "syncg_test_combined_severe.jsonl"
        ),
        "shared_predictions_sha256": (
            "1014832d6e401e983cc345606d18ca3190e3f60818ca0926a2bce90e1d750bc3"
        ),
        "shared_predictions_metadata_sha256": (
            "5f29a58679899d7466c7fabc0a4e29eb0008e2bc83bc2c400aa7d9e7f09eed26"
        ),
        "condition": "combined_severe",
    },
    "rpm10k": {
        "manifest": _RPM_MANIFEST,
        "manifest_sha256": _RPM_MANIFEST_SHA256,
        "manifest_protocol_sha256": _RPM_PROTOCOL_SHA256,
        "shared_predictions": (
            "artifacts/predictions/rpm10k_single_pointer_test.jsonl"
        ),
        "shared_predictions_sha256": (
            "d2ce3a0fc6df049c741b2f6f92fc382b67bbb54f74d01872b0c4e51a1931e8d7"
        ),
        "shared_predictions_metadata_sha256": (
            "3cdd22b50ab91dd66dbe07e289caf58ab7ad6a8be8907dc603dbd96e23185d85"
        ),
        "condition": "clean",
    },
}


def evaluation_plan_source_sha256() -> str:
    return sha256_source_file(Path(__file__).resolve())


def evaluation_execution_source_identity() -> dict[str, Any]:
    relative_paths = {
        "evaluator": "experiments/evaluate_vdn_baseline.py",
        "vdn_adapter": "experiments/vdn_baseline.py",
        "dataset_protocol": "experiments/datasets.py",
        "phase2_protocol": "experiments/vdn_phase2_protocol.py",
        "formal_environment_gate": (
            "experiments/formal_environment.py"
        ),
        "degradation": "experiments/robustness_degradations.py",
        "meter_detector": (
            "utils/angleDetect/yoloDetection/yoloDectect.py"
        ),
        "reference_point_geometry": (
            "utils/angleDetect/yoloDetection/pointGet.py"
        ),
        "isolated_bootstrap": (
            "experiments/run_vdn_phase2_evaluation_isolated.py"
        ),
        "aggregation_isolated_bootstrap": (
            "experiments/run_vdn_phase2_aggregation_isolated.py"
        ),
        "evaluation_supervisor": (
            "experiments/run_vdn_phase2_evaluations.ps1"
        ),
        "replicate_aggregator": (
            "experiments/summarize_vdn_phase2_replicates.py"
        ),
        "shared_aggregation_helpers": (
            "experiments/summarize_vdn_replicates.py"
        ),
        "public_preflight": (
            "experiments/preflight_vdn_phase2_evaluations.py"
        ),
        "public_preflight_isolated_bootstrap": (
            "experiments/run_vdn_phase2_preflight_isolated.py"
        ),
    }
    environment = formal_environment_identity(
        PROJECT_DIR,
        validate_process=False,
    )
    return {
        "protocol": PHASE2_EVALUATION_EXECUTION_SOURCE_PROTOCOL,
        "source_hash_protocol": (
            PHASE2_EVALUATION_PLAN_SOURCE_SHA256_PROTOCOL
        ),
        "files": {
            name: sha256_source_file(PROJECT_DIR / relative)
            for name, relative in sorted(relative_paths.items())
        },
        "environment_lock": {
            "protocol": "raw_file_sha256_v1",
            "path": environment["requirements_lock_path"],
            "sha256": environment["requirements_lock_sha256"],
        },
        "installed_environment": environment,
    }


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise PermissionError(
            f"Phase-2 evaluation plan mismatch for {label}: "
            f"expected {expected!r}, got {actual!r}"
        )


def phase2_evaluation_identity(
    name: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Return the exact, read-only identity for one authorized public plan."""

    run_dir = Path(run_dir).resolve()
    if run_dir.parent != FORMAL_RUN_ROOT.resolve():
        raise PermissionError("Phase-2 run is outside the formal run root")
    plan = PHASE2_EVALUATION_PLANS.get(str(name))
    if plan is None:
        raise PermissionError(
            f"Phase-2 evaluation {name!r} is not in the public plan"
        )
    manifest = _project_path(str(plan["manifest"]))
    protocol = manifest.with_name(manifest.name + ".protocol.json")
    shared = _project_path(str(plan["shared_predictions"]))
    shared_metadata = shared.with_name(shared.name + ".meta.json")
    return {
        "protocol": PHASE2_EVALUATION_PLAN_PROTOCOL,
        "plan_source_sha256_protocol": (
            PHASE2_EVALUATION_PLAN_SOURCE_SHA256_PROTOCOL
        ),
        "plan_source_sha256": evaluation_plan_source_sha256(),
        "execution_source_identity": (
            evaluation_execution_source_identity()
        ),
        "name": str(name),
        "manifest_path": str(manifest),
        "manifest_sha256": str(plan["manifest_sha256"]),
        "manifest_protocol_path": str(protocol),
        "manifest_protocol_sha256": str(
            plan["manifest_protocol_sha256"]
        ),
        "shared_predictions_path": str(shared),
        "shared_predictions_sha256": str(
            plan["shared_predictions_sha256"]
        ),
        "shared_predictions_metadata_path": str(shared_metadata),
        "shared_predictions_metadata_sha256": str(
            plan["shared_predictions_metadata_sha256"]
        ),
        "vdn_source_path": str(FORMAL_VDN_SOURCE.resolve()),
        "meter_detector_weights_path": str(
            FORMAL_METER_WEIGHTS.resolve()
        ),
        "meter_detector_weights_sha256": (
            FORMAL_METER_WEIGHTS_SHA256
        ),
        "keypoint_detector_weights_path": str(
            FORMAL_POINT_WEIGHTS.resolve()
        ),
        "keypoint_detector_weights_sha256": (
            FORMAL_POINT_WEIGHTS_SHA256
        ),
        "condition": str(plan["condition"]),
        "degradation_seed": FORMAL_DEGRADATION_SEED,
        "device": "cuda",
        "amp_enabled": True,
        "batch_size": FORMAL_BATCH_SIZE,
        "bootstrap_iterations": FORMAL_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
        "output_path": str(run_dir / "evaluations" / f"{name}.jsonl"),
        "field_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
    }


def validate_phase2_evaluation_plan(
    args: Any,
    checkpoint_path: Path,
    *,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, Any]:
    """Authorize one exact public plan tuple after cohort validation."""

    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent
    if (
        checkpoint_path != run_dir / "best.pt"
        or run_dir.parent != FORMAL_RUN_ROOT.resolve()
    ):
        raise PermissionError("Phase-2 checkpoint is outside the formal run root")
    output = Path(args.output).resolve()
    expected_output_root = run_dir / "evaluations"
    if output.parent != expected_output_root or output.suffix != ".jsonl":
        raise PermissionError(
            "Phase-2 output is outside the formal supporting-evaluation root"
        )
    name = output.stem
    plan = PHASE2_EVALUATION_PLANS.get(name)
    if plan is None:
        raise PermissionError(
            f"Phase-2 output name {name!r} is not in the public plan"
        )

    expected_manifest = _project_path(str(plan["manifest"]))
    expected_protocol = expected_manifest.with_name(
        expected_manifest.name + ".protocol.json"
    )
    expected_shared = _project_path(str(plan["shared_predictions"]))
    expected_shared_metadata = expected_shared.with_name(
        expected_shared.name + ".meta.json"
    )
    shape_checks = (
        (Path(args.manifest).resolve(), expected_manifest, "manifest path"),
        (
            (
                Path(args.shared_predictions).resolve()
                if args.shared_predictions is not None
                else None
            ),
            expected_shared,
            "shared-predictions path",
        ),
        (Path(args.vdn_source).resolve(), FORMAL_VDN_SOURCE.resolve(), "VDN source"),
        (
            Path(args.meter_detector_weights).resolve(),
            FORMAL_METER_WEIGHTS.resolve(),
            "meter detector weights path",
        ),
        (
            Path(args.keypoint_detector_weights).resolve(),
            FORMAL_POINT_WEIGHTS.resolve(),
            "keypoint detector weights path",
        ),
        (str(args.condition), str(plan["condition"]), "condition"),
        (
            int(args.degradation_seed),
            FORMAL_DEGRADATION_SEED,
            "degradation seed",
        ),
        (str(args.device), "cuda", "device"),
        (bool(args.no_amp), False, "no-AMP flag"),
        (int(args.batch_size), FORMAL_BATCH_SIZE, "batch size"),
        (
            int(args.bootstrap_iterations),
            FORMAL_BOOTSTRAP_ITERATIONS,
            "bootstrap iterations",
        ),
        (int(args.seed), FORMAL_BOOTSTRAP_SEED, "bootstrap seed"),
        (args.limit, None, "diagnostic limit"),
        (bool(args.overwrite), False, "overwrite flag"),
    )
    for actual, expected, label in shape_checks:
        _require_equal(actual, expected, label=label)

    for path, label in (
        (expected_manifest, "manifest"),
        (expected_protocol, "manifest protocol"),
        (expected_shared, "shared predictions"),
        (expected_shared_metadata, "shared prediction metadata"),
        (FORMAL_METER_WEIGHTS.resolve(), "meter detector weights"),
        (FORMAL_POINT_WEIGHTS.resolve(), "keypoint detector weights"),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    hash_checks = (
        (
            hash_file(expected_manifest),
            str(plan["manifest_sha256"]),
            "manifest SHA-256",
        ),
        (
            hash_file(expected_protocol),
            str(plan["manifest_protocol_sha256"]),
            "manifest protocol SHA-256",
        ),
        (
            hash_file(expected_shared),
            str(plan["shared_predictions_sha256"]),
            "shared predictions SHA-256",
        ),
        (
            hash_file(expected_shared_metadata),
            str(plan["shared_predictions_metadata_sha256"]),
            "shared prediction metadata SHA-256",
        ),
        (
            hash_file(FORMAL_METER_WEIGHTS.resolve()),
            FORMAL_METER_WEIGHTS_SHA256,
            "meter detector weights SHA-256",
        ),
        (
            hash_file(FORMAL_POINT_WEIGHTS.resolve()),
            FORMAL_POINT_WEIGHTS_SHA256,
            "keypoint detector weights SHA-256",
        ),
    )
    for actual, expected, label in hash_checks:
        _require_equal(actual, expected, label=label)

    identity = phase2_evaluation_identity(name, run_dir)
    _require_equal(
        identity["output_path"],
        str(output),
        label="evaluation output identity",
    )
    return identity


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path} contains non-finite JSON constant {value}")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path} repeats JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def validate_phase2_public_preflight(
    preflight_path: Path,
    cohort_authorization: Path,
    *,
    expected_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the exact formal public preflight without creating a cycle."""

    preflight_path = Path(preflight_path).resolve()
    cohort_authorization = Path(cohort_authorization).resolve()
    if preflight_path != FORMAL_PUBLIC_PREFLIGHT.resolve():
        raise PermissionError("Phase-2 public preflight path is not formal")
    if not preflight_path.is_file():
        raise FileNotFoundError(preflight_path)
    report = _strict_json(preflight_path)
    expected_keys = {
        "schema_version",
        "protocol",
        "status",
        "cohort_authorization_path",
        "cohort_authorization_sha256",
        "evaluation_plan_source_sha256",
        "evaluation_execution_source_identity",
        "parent_seeds",
        "plans",
        "public_release_inventory_sha256",
        "frozen_final_artifacts",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
    }
    if set(report) != expected_keys:
        raise ValueError("Phase-2 public preflight schema drifted")
    if (
        report.get("schema_version")
        != PHASE2_PUBLIC_PREFLIGHT_SCHEMA_VERSION
        or report.get("protocol") != PHASE2_PUBLIC_PREFLIGHT_PROTOCOL
        or report.get("status") != "passed"
        or Path(
            str(report.get("cohort_authorization_path") or "")
        ).resolve()
        != cohort_authorization
        or report.get("cohort_authorization_sha256")
        != sha256_file(cohort_authorization)
        or report.get("evaluation_plan_source_sha256")
        != evaluation_plan_source_sha256()
        or report.get("evaluation_execution_source_identity")
        != evaluation_execution_source_identity()
        or report.get("parent_seeds") != list(FORMAL_PARENT_SEEDS)
        or report.get("public_release_inventory_sha256")
        != PINNED_PUBLIC_RELEASE_INVENTORY_SHA256
        or report.get("public_data_opened_or_read") is not True
        or report.get("field_data_opened_or_read") is not False
        or report.get("sealed_data_opened_or_read") is not False
    ):
        raise ValueError("Phase-2 public preflight authorization drifted")

    expected_plans = {
        str(seed): {
            name: phase2_evaluation_identity(
                name,
                FORMAL_RUN_ROOT / f"seed_{seed}",
            )
            for name in PHASE2_EVALUATION_PLANS
        }
        for seed in FORMAL_PARENT_SEEDS
    }
    if report.get("plans") != expected_plans:
        raise ValueError("Phase-2 public preflight plan matrix drifted")
    if expected_plan is not None:
        run_dir = Path(str(expected_plan["output_path"])).resolve().parent.parent
        try:
            parent_seed = int(run_dir.name.removeprefix("seed_"))
            recorded = report["plans"][str(parent_seed)][
                str(expected_plan["name"])
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Phase-2 public preflight lacks the requested plan"
            ) from exc
        if recorded != dict(expected_plan):
            raise ValueError(
                "Phase-2 public preflight requested plan drifted"
            )

    final_artifacts = report.get("frozen_final_artifacts")
    if (
        not isinstance(final_artifacts, Mapping)
        or set(final_artifacts)
        != set(PINNED_FINAL_EVALUATION_ARTIFACT_SHA256)
    ):
        raise ValueError(
            "Phase-2 public preflight final-artifact set drifted"
        )
    for condition, pins in (
        PINNED_FINAL_EVALUATION_ARTIFACT_SHA256.items()
    ):
        record = final_artifacts.get(condition)
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "prediction_path",
                "prediction_sha256",
                "summary_path",
                "summary_sha256",
            }
            or record.get("prediction_sha256") != pins["predictions"]
            or record.get("summary_sha256") != pins["summary"]
        ):
            raise ValueError(
                f"Phase-2 public preflight {condition} final pin drifted"
            )
    return {
        "protocol": PHASE2_PUBLIC_PREFLIGHT_PROTOCOL,
        "schema_version": PHASE2_PUBLIC_PREFLIGHT_SCHEMA_VERSION,
        "path": str(preflight_path),
        "sha256": sha256_file(preflight_path),
        "cohort_authorization_sha256": sha256_file(
            cohort_authorization
        ),
    }


__all__ = [
    "FORMAL_PUBLIC_PREFLIGHT",
    "FORMAL_RUN_ROOT",
    "PHASE2_PUBLIC_PREFLIGHT_PROTOCOL",
    "PHASE2_PUBLIC_PREFLIGHT_SCHEMA_VERSION",
    "PHASE2_EVALUATION_PLAN_PROTOCOL",
    "PHASE2_EVALUATION_PLANS",
    "PINNED_FINAL_EVALUATION_ARTIFACT_SHA256",
    "PINNED_PUBLIC_RELEASE_INVENTORY_SHA256",
    "evaluation_execution_source_identity",
    "evaluation_plan_source_sha256",
    "phase2_evaluation_identity",
    "validate_phase2_evaluation_plan",
    "validate_phase2_public_preflight",
]
