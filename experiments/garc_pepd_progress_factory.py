"""Frozen PEPD progress factory and binding writer for formal GARC runs.

The factory resolves one of the three authoritative grouped-holdout PEPD
checkpoints from the already authenticated handoff.  It never accepts an
image, label, range value, or field namespace.  ``freeze-binding`` constructs
the provider identity without running inference and writes an immutable
``FrozenComponentBinding`` for later plan freezing.
"""
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
from experiments.run_cagh_v5_enhanced_oof import HANDOFF, PEPD_SEEDS
from experiments.v5_unified_full_auto_adapter import FrozenComponentBinding
from experiments.v5_unified_full_auto_progress_providers import (
    PEPDFullAutoProgressProvider,
    PROTOCOL as PROGRESS_PROVIDER_PROTOCOL,
)


FACTORY_PROTOCOL: Final[str] = "garc_authoritative_pepd_progress_factory_v1"
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


def _authoritative_rows() -> dict[int, dict[str, Any]]:
    handoff = strict_json(HANDOFF)
    require(
        handoff.get("status") == "authorized"
        and handoff.get("protocol") == "pepd_mixed_authoritative_oof_handoff_v2",
        "authoritative PEPD handoff is not ready",
    )
    values = handoff.get("authoritative_runs")
    require(isinstance(values, Mapping), "authoritative PEPD runs are absent")
    result: dict[int, dict[str, Any]] = {}
    for seed in PEPD_SEEDS:
        row = values.get(str(seed))
        require(isinstance(row, Mapping), f"PEPD seed {seed} handoff is absent")
        checkpoint = Path(str(row.get("authoritative_best_checkpoint") or "")).resolve(
            strict=True
        )
        verification = Path(str(row.get("verification") or "")).resolve(strict=True)
        require(
            sha256_file(checkpoint)
            == row.get("authoritative_best_checkpoint_sha256"),
            f"PEPD seed {seed} checkpoint hash drift",
        )
        require(
            sha256_file(verification) == row.get("verification_sha256"),
            f"PEPD seed {seed} verification hash drift",
        )
        result[int(seed)] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "verification": verification,
            "verification_sha256": sha256_file(verification),
        }
    return result


def _seed_from_plan(plan: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    artifacts = plan["progress_component"]["binding"]["artifact_sha256"]
    checkpoint_sha = str(artifacts.get("checkpoint") or "").casefold()
    rows = _authoritative_rows()
    matches = [
        (seed, row)
        for seed, row in rows.items()
        if row["checkpoint_sha256"] == checkpoint_sha
    ]
    require(len(matches) == 1, "plan does not bind one authoritative PEPD checkpoint")
    return matches[0]


def _provider(
    row: Mapping[str, Any],
    *,
    device: str,
    amp_enabled: bool,
) -> PEPDFullAutoProgressProvider:
    detector = REFERENCE_DETECTOR.resolve(strict=True)
    return PEPDFullAutoProgressProvider.from_frozen_files(
        checkpoint_path=Path(row["checkpoint"]),
        expected_checkpoint_sha256=str(row["checkpoint_sha256"]),
        verification_path=Path(row["verification"]),
        reference_detector_path=detector,
        expected_reference_detector_sha256=sha256_file(detector),
        device=device,
        amp_enabled=amp_enabled,
    )


def build_progress_provider(plan: Mapping[str, Any]) -> PEPDFullAutoProgressProvider:
    """Runtime entrypoint bound by ``garc_full_auto_public.freeze-plan``."""

    seed, row = _seed_from_plan(plan)
    del seed
    declared_detector = str(plan["reference"].get("detector_sha256") or "")
    require(
        declared_detector == sha256_file(REFERENCE_DETECTOR),
        "plan reference detector differs from authoritative factory",
    )
    device = str(plan["garc"]["device"])
    return _provider(
        row,
        device=device,
        amp_enabled=device.casefold().startswith("cuda"),
    )


def freeze_binding(*, seed: int, output: Path, runtime_device: str) -> Path:
    require(int(seed) in PEPD_SEEDS, "PEPD seed is not authoritative")
    row = _authoritative_rows()[int(seed)]
    # Identity is device-invariant except for the explicit AMP flag.  Building
    # on CPU avoids reserving GPU memory during plan preparation.
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
            "authoritative_handoff": sha256_file(HANDOFF),
        },
        synthetic=False,
        verified_complete=True,
    )
    target = guard_public_path(
        output, label="GARC PEPD progress binding", must_exist=False
    )
    require(not target.exists(), f"refusing to overwrite progress binding: {target}")
    binding.write(target)
    return target


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-binding")
    freeze.add_argument("--seed", type=int, choices=PEPD_SEEDS, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--runtime-device", default="cuda:0")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--seed", type=int, choices=PEPD_SEEDS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "freeze-binding":
        output = freeze_binding(
            seed=args.seed,
            output=args.output,
            runtime_device=args.runtime_device,
        )
        payload = {"binding": str(output), "sha256": sha256_file(output)}
    else:
        row = _authoritative_rows()[int(args.seed)]
        payload = {
            "protocol": FACTORY_PROTOCOL,
            "seed": int(args.seed),
            "checkpoint": str(row["checkpoint"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "verification": str(row["verification"]),
            "verification_sha256": row["verification_sha256"],
            "reference_detector": str(REFERENCE_DETECTOR),
            "reference_detector_sha256": sha256_file(REFERENCE_DETECTOR),
            "images_opened": 0,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "FACTORY_PROTOCOL",
    "REFERENCE_DETECTOR",
    "build_progress_provider",
    "freeze_binding",
]
