"""Frozen official-200 VDN progress factory for GARC OOF comparisons."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.automatic_numeric_range_public_protocol import (
    guard_public_path,
    require,
    sha256_file,
    strict_json,
)
from experiments.vdn_baseline import verify_vdn_source
from experiments.v5_unified_full_auto_adapter import FrozenComponentBinding
from experiments.v5_unified_full_auto_progress_providers import (
    PROTOCOL as PROGRESS_PROVIDER_PROTOCOL,
    VDNFullAutoProgressProvider,
)


FACTORY_PROTOCOL: Final[str] = "garc_vdn_official200_progress_factory_v1"
OOF_PROTOCOL: Final[str] = "vdn_official200_same_sample_train_grouped_oof_evaluation_v1"
SEEDS: Final[tuple[int, ...]] = (20260720, 20260721, 20260722)
OOF_SUMMARY: Final[Path] = (
    _PROJECT_ROOT / "artifacts/runs/vdn_official200_train_oof_v1/summary.json"
).resolve()
VDN_SOURCE: Final[Path] = (
    _PROJECT_ROOT / "artifacts/vendor/VectorDetectionNetwork"
).resolve()
REFERENCE_DETECTOR: Final[Path] = (
    _PROJECT_ROOT / "utils/angleDetect/yoloDetection/result/yolo_pointbest.pt"
).resolve()
SOURCE: Final[Path] = Path(__file__).resolve()
WRAPPER_SOURCE: Final[Path] = SOURCE.with_name(
    "v5_unified_full_auto_progress_providers.py"
)
DIRECTION_ADAPTER_SOURCE: Final[Path] = SOURCE.with_name(
    "v5_unified_direction_adapters.py"
)


def authoritative_rows() -> dict[int, dict[str, Any]]:
    summary = strict_json(OOF_SUMMARY)
    require(summary.get("protocol") == OOF_PROTOCOL, "VDN OOF protocol drift")
    require(summary.get("status") == "complete", "VDN OOF summary incomplete")
    cohort = summary.get("cohort") or {}
    require(
        int(cohort.get("samples", 0)) == 4380
        and int(cohort.get("physical_groups", 0)) == 197
        and int(cohort.get("group_leakage_count", -1)) == 0,
        "VDN OOF cohort drift",
    )
    source_commit = verify_vdn_source(VDN_SOURCE)
    values = summary.get("bindings", {}).get("checkpoint_by_seed") or {}
    result: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        row = values.get(str(seed))
        require(isinstance(row, Mapping), f"VDN seed {seed} binding absent")
        checkpoint = guard_public_path(
            _PROJECT_ROOT / str(row.get("path") or ""),
            label=f"VDN seed {seed} checkpoint",
        )
        require(
            sha256_file(checkpoint) == row.get("sha256"),
            f"VDN seed {seed} checkpoint hash drift",
        )
        verification = guard_public_path(
            checkpoint.parent / "verification_v1.json",
            label=f"VDN seed {seed} verification",
        )
        verified = strict_json(verification)
        require(
            verified.get("protocol") == "formal_vdn_official200_verification_v1"
            and verified.get("verified") is True
            and int(verified.get("seed", -1)) == seed
            and verified.get("eligible_for_three_seed_cohort") is True
            and verified.get("official_stopping_boundary_reached") is True,
            f"VDN seed {seed} verification is ineligible",
        )
        require(
            verified.get("best_checkpoint_sha256") == sha256_file(checkpoint),
            f"VDN seed {seed} verification/checkpoint drift",
        )
        for key in (
            "field_data_opened_or_read",
            "test_data_opened_or_read",
            "sealed_data_opened_or_read",
            "confirmatory_data_opened_or_read",
        ):
            require(verified.get(key) is False, f"VDN seed {seed} scope violation: {key}")
        run_summary = strict_json(checkpoint.parent / "summary.json")
        signature = run_summary.get("signature") or {}
        require(
            signature.get("vdn_source_commit") == source_commit,
            f"VDN seed {seed} source commit drift",
        )
        result[seed] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "verification": verification,
            "verification_sha256": sha256_file(verification),
            "validation_samples": int(row["validation_samples"]),
            "validation_groups": int(row["validation_groups"]),
            "validation_sample_ids_sha256": str(row["validation_sample_ids_sha256"]),
            "vdn_source_commit": source_commit,
        }
    return result


def _seed_from_plan(plan: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    checkpoint_sha = str(
        plan["progress_component"]["binding"]["artifact_sha256"].get("checkpoint")
        or ""
    ).casefold()
    matches = [
        (seed, row)
        for seed, row in authoritative_rows().items()
        if row["checkpoint_sha256"] == checkpoint_sha
    ]
    require(len(matches) == 1, "plan does not bind one official-200 VDN checkpoint")
    return matches[0]


def _provider(
    row: Mapping[str, Any], *, device: str, amp_enabled: bool
) -> VDNFullAutoProgressProvider:
    return VDNFullAutoProgressProvider.from_frozen_files(
        checkpoint_path=Path(row["checkpoint"]),
        expected_checkpoint_sha256=str(row["checkpoint_sha256"]),
        verification_path=Path(row["verification"]),
        vdn_source=VDN_SOURCE,
        reference_detector_path=REFERENCE_DETECTOR,
        expected_reference_detector_sha256=sha256_file(REFERENCE_DETECTOR),
        device=device,
        amp_enabled=amp_enabled,
    )


def build_progress_provider(plan: Mapping[str, Any]) -> VDNFullAutoProgressProvider:
    _, row = _seed_from_plan(plan)
    require(
        str(plan["reference"].get("detector_sha256") or "")
        == sha256_file(REFERENCE_DETECTOR),
        "plan reference detector differs from VDN factory",
    )
    device = str(plan["garc"]["device"])
    return _provider(
        row,
        device=device,
        amp_enabled=device.casefold().startswith("cuda"),
    )


def freeze_binding(*, seed: int, output: Path, runtime_device: str) -> Path:
    require(int(seed) in SEEDS, "VDN seed is not authoritative")
    row = authoritative_rows()[int(seed)]
    provider = _provider(
        row,
        device="cpu",
        amp_enabled=str(runtime_device).casefold().startswith("cuda"),
    )
    binding = FrozenComponentBinding.from_provider(
        name="progress",
        provider=provider,
        provider_protocol=PROGRESS_PROVIDER_PROTOCOL,
        artifact_sha256={
            "checkpoint": row["checkpoint_sha256"],
            "verification": row["verification_sha256"],
            "reference_detector": sha256_file(REFERENCE_DETECTOR),
        },
        source_sha256={
            "factory": sha256_file(SOURCE),
            "progress_wrapper": sha256_file(WRAPPER_SOURCE),
            "direction_adapter": sha256_file(DIRECTION_ADAPTER_SOURCE),
            "oof_summary": sha256_file(OOF_SUMMARY),
        },
        synthetic=False,
        verified_complete=True,
    )
    target = guard_public_path(output, label="GARC VDN progress binding", must_exist=False)
    require(not target.exists(), f"refusing to overwrite VDN binding: {target}")
    binding.write(target)
    return target


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-binding")
    freeze.add_argument("--seed", type=int, choices=SEEDS, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--runtime-device", default="cuda:0")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--seed", type=int, choices=SEEDS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    row = authoritative_rows()[int(args.seed)]
    if args.command == "freeze-binding":
        path = freeze_binding(
            seed=args.seed, output=args.output, runtime_device=args.runtime_device
        )
        value = {"binding": str(path), "sha256": sha256_file(path)}
    else:
        value = {
            "protocol": FACTORY_PROTOCOL,
            "seed": int(args.seed),
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in row.items()
            },
            "images_opened": 0,
        }
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "FACTORY_PROTOCOL",
    "OOF_SUMMARY",
    "SEEDS",
    "VDN_SOURCE",
    "authoritative_rows",
    "build_progress_provider",
    "freeze_binding",
]
