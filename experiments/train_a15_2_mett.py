"""Train the A15.2 METT correction on the existing physical triplets.

The imported EfficientNet-B0 point anchor is frozen.  Only the endpoint-
tangent correction is optimized; the moment-exact posterior lift therefore
retains the source scalar prediction while supplying the distributional state
required by FTEB.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments.a15_2_mett import (
    A15_2_METT_ARCHITECTURE,
    A152METTCorrection,
    DIRECT_BASELINE_EPOCHS,
    MomentExactEfficientNetB0Anchor,
    ReMSTCorrection,
    load_moment_exact_anchor_from_direct_checkpoint,
    remst_publication_identity,
)
from experiments.a15_fteb import fteb_parameter_counts
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    load_a13_correction_train_manifest,
)
from experiments.train_a15_2_fteb_correction_only import (
    INITIALIZATION_SEED,
    LEARNING_RATE,
    SAMPLE_ORDER_SEED,
    TERMINAL_EPOCHS,
    WEIGHT_DECAY,
    build_a15_2_epoch_loader,
    build_a15_2_train_dataset,
    configure_a15_2_reproducibility,
    run_a15_2_correction_epoch,
)


PROTOCOL: Final[str] = "syncg_a15_2_mett_correction_experiment_v1"
FULL_METT_VARIANT: Final[dict[str, bool]] = {
    "use_tangent": True,
    "use_shape": True,
    "use_direct_scalar_residual": False,
}
DIRECT_SCALAR_METT_VARIANT: Final[dict[str, bool]] = {
    "use_tangent": False,
    "use_shape": False,
    "use_direct_scalar_residual": True,
}
METT_VARIANTS: Final[dict[str, dict[str, bool]]] = {
    "full": FULL_METT_VARIANT,
    "direct_scalar": DIRECT_SCALAR_METT_VARIANT,
}
FULL_METT_CONSTRUCTION_FIELDS: Final[tuple[str, ...]] = (
    "relation_channels",
    "token_dim",
    "attention_heads",
    "decoder_layers",
    "memory_grid_size",
    "progress_bins",
)
DEFAULT_DIRECT_CHECKPOINT: Final[Path] = Path(
    "C:/pointer_read/paper_syncg_only_retrain_v1/lightweight_baselines/"
    "efficientnet_b0/seed_20262022/terminal.pt"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def publication_model_identity(expected_variant: str) -> dict[str, str]:
    """Return a human-facing paper identity for a legacy experiment variant."""

    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    if expected_variant == "direct_scalar":
        return remst_publication_identity()
    return {
        "short_name": "METT",
        "full_name": "Moment-Exact Endpoint-Tangent Transport",
        "display_name": "METT (full)",
        "architecture": A15_2_METT_ARCHITECTURE,
        "legacy_family": "A15.2-METT",
        "legacy_experiment_variant": "full",
        "legacy_checkpoint_architecture": A15_2_METT_ARCHITECTURE,
    }


def validate_mett_variant_metadata(
    metadata: Mapping[str, Any], *, expected_variant: str
) -> dict[str, Any]:
    """Return the canonical identity of one explicit terminal METT variant."""

    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    expected_flags = METT_VARIANTS[expected_variant]
    _require(bool(str(metadata.get("checkpoint", "")).strip()), "METT checkpoint is missing")
    _require(int(metadata.get("epochs", -1)) == TERMINAL_EPOCHS, "METT run is not terminal")
    construction = metadata.get("construction")
    _require(isinstance(construction, Mapping), "METT construction is missing")
    variant_source = metadata.get("experiment_variant")
    if not isinstance(variant_source, Mapping):
        variant_source = construction
    variant = {
        name: bool(variant_source.get(name, default))
        for name, default in FULL_METT_VARIANT.items()
    }
    _require(
        variant == expected_flags,
        f"METT checkpoint is not the requested {expected_variant} variant",
    )
    _require(
        not metadata.get("inference_sensitivity_overrides"),
        "METT metadata contains inference sensitivity overrides",
    )
    parameter_counts = metadata.get("parameter_counts")
    training_seeds = metadata.get("seeds")
    _require(
        all(name in construction for name in FULL_METT_CONSTRUCTION_FIELDS)
        and isinstance(parameter_counts, Mapping)
        and all(name in parameter_counts for name in ("anchor", "correction", "trainable"))
        and isinstance(training_seeds, Mapping)
        and all(name in training_seeds for name in ("initialization", "sample_order")),
        "METT terminal-run identity is incomplete",
    )
    return {
        "checkpoint": str(metadata["checkpoint"]),
        "epochs": TERMINAL_EPOCHS,
        "variant_label": expected_variant,
        "experiment_variant": variant,
        "construction": {
            name: int(construction[name]) for name in FULL_METT_CONSTRUCTION_FIELDS
        },
        "parameter_counts": {
            name: int(parameter_counts[name])
            for name in ("anchor", "correction", "trainable")
        },
        "training_seeds": {
            name: int(training_seeds[name])
            for name in ("initialization", "sample_order")
        },
        "publication_model": publication_model_identity(expected_variant),
    }


def validate_full_mett_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical identity of a terminal, non-ablation full METT run."""

    return validate_mett_variant_metadata(metadata, expected_variant="full")


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def load_mett_models(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[torch.nn.Module, A152METTCorrection, dict[str, Any]]:
    """Load a METT training artifact for evaluation or continuation."""

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"METT checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("protocol") == PROTOCOL
        and checkpoint.get("architecture") == A15_2_METT_ARCHITECTURE,
        "METT checkpoint metadata differs",
    )
    direct = checkpoint.get("source_direct_checkpoint")
    _require(isinstance(direct, str) and bool(direct), "METT source checkpoint missing")
    anchor_state = checkpoint.get("anchor_state")
    correction_state = checkpoint.get("correction_state")
    _require(
        isinstance(anchor_state, Mapping)
        and isinstance(correction_state, Mapping),
        "METT checkpoint model states are missing",
    )
    construction = checkpoint.get("construction")
    stored_source_metadata = checkpoint.get("source_anchor")
    _require(isinstance(construction, Mapping), "METT construction is missing")
    _require(
        isinstance(stored_source_metadata, Mapping),
        "METT stored source-anchor metadata is missing",
    )
    _require(
        str(Path(str(stored_source_metadata.get("source", ""))).resolve())
        == str(Path(direct).resolve())
        and stored_source_metadata.get("source_protocol")
        == "syncg_lightweight_regression_baselines_v1"
        and stored_source_metadata.get("source_architecture")
        == "efficientnet_b0"
        and int(stored_source_metadata.get("source_epochs", -1))
        == DIRECT_BASELINE_EPOCHS
        and int(stored_source_metadata.get("progress_bins", -1))
        == int(construction.get("progress_bins", -2)),
        "METT stored source-anchor identity differs",
    )
    target_device = torch.device(device)
    anchor = MomentExactEfficientNetB0Anchor(
        progress_bins=int(stored_source_metadata["progress_bins"]),
        initial_scale=float(
            stored_source_metadata["initial_posterior_scale"]
        ),
    ).to(target_device)
    anchor_load = anchor.load_state_dict(anchor_state, strict=True)
    _require(
        not anchor_load.missing_keys and not anchor_load.unexpected_keys,
        "METT anchor state does not load strictly",
    )
    construction_flags = {
        name: bool(construction.get(name, default))
        for name, default in FULL_METT_VARIANT.items()
    }
    if construction_flags == DIRECT_SCALAR_METT_VARIANT:
        remst_construction: dict[str, Any] = {
            name: int(construction[name])
            for name in FULL_METT_CONSTRUCTION_FIELDS
        }
        if "use_relation_memory" in construction:
            remst_construction["use_relation_memory"] = bool(
                construction["use_relation_memory"]
            )
        correction = ReMSTCorrection(**remst_construction).to(target_device)
    else:
        correction = A152METTCorrection(
            **{name: value for name, value in construction.items()}
        ).to(target_device)
    correction_load = correction.load_state_dict(correction_state, strict=True)
    _require(
        not correction_load.missing_keys
        and not correction_load.unexpected_keys,
        "METT correction state does not load strictly",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    correction.eval()
    variant = checkpoint.get("experiment_variant")
    if not isinstance(variant, Mapping):
        variant = {
            "use_tangent": bool(construction.get("use_tangent", True)),
            "use_shape": bool(construction.get("use_shape", True)),
            "use_direct_scalar_residual": bool(
                construction.get("use_direct_scalar_residual", False)
            ),
        }
    else:
        for name, default in FULL_METT_VARIANT.items():
            if name in construction:
                _require(
                    bool(variant.get(name, default)) == bool(construction[name]),
                    f"METT experiment variant and construction differ: {name}",
                )
    model_metadata: dict[str, Any] = {
        "checkpoint": str(source),
        "epochs": int(checkpoint["epochs"]),
        "source_anchor": dict(stored_source_metadata),
        "parameter_counts": checkpoint.get("parameter_counts"),
        "construction": dict(construction),
        "experiment_variant": dict(variant),
        "seeds": checkpoint.get("seeds"),
        "training_elapsed_seconds": float(
            checkpoint.get("training_elapsed_seconds", 0.0)
        ),
    }
    canonical_variant = {
        name: bool(variant.get(name, default))
        for name, default in FULL_METT_VARIANT.items()
    }
    if canonical_variant == DIRECT_SCALAR_METT_VARIANT:
        model_metadata["publication_model"] = publication_model_identity(
            "direct_scalar"
        )
    elif canonical_variant == FULL_METT_VARIANT:
        model_metadata["publication_model"] = publication_model_identity("full")
    return anchor, correction, model_metadata


def train_mett(
    *,
    correction_train_manifest_path: Path,
    direct_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    epochs: int = TERMINAL_EPOCHS,
    initialization_seed: int = INITIALIZATION_SEED,
    sample_order_seed: int = SAMPLE_ORDER_SEED,
    use_tangent: bool = True,
    use_shape: bool = True,
    use_direct_scalar_residual: bool = False,
) -> dict[str, Any]:
    _require(workers >= 0, "METT workers must be non-negative")
    _require(1 <= int(epochs) <= TERMINAL_EPOCHS, "METT epochs must be in [1, 5]")
    _require(int(initialization_seed) >= 0, "METT initialization seed is negative")
    _require(int(sample_order_seed) >= 0, "METT sample-order seed is negative")
    _require(
        bool(use_tangent)
        or bool(use_shape)
        or bool(use_direct_scalar_residual),
        "METT has no learned mechanism",
    )
    _require(
        not bool(use_direct_scalar_residual)
        or (not bool(use_tangent) and not bool(use_shape)),
        "direct-scalar control cannot also enable METT tangent or shape",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"METT output already exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "METT device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_a15_2_reproducibility(device, seed=int(sample_order_seed))

    samples = tuple(
        load_a13_correction_train_manifest(
            Path(correction_train_manifest_path).resolve()
        )
    )
    dataset = build_a15_2_train_dataset(samples)
    anchor, anchor_metadata = load_moment_exact_anchor_from_direct_checkpoint(
        direct_checkpoint_path, device=device
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    anchor.eval()
    anchor_initial_state = _state_cpu(anchor)

    torch.manual_seed(int(initialization_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(initialization_seed))
    construction = {
        "relation_channels": 48,
        "token_dim": 64,
        "attention_heads": 4,
        "decoder_layers": 2,
        "memory_grid_size": 4,
        "progress_bins": 128,
        "use_tangent": bool(use_tangent),
        "use_shape": bool(use_shape),
        "use_direct_scalar_residual": bool(use_direct_scalar_residual),
    }
    if bool(use_direct_scalar_residual):
        correction = ReMSTCorrection(
            **{
                name: int(construction[name])
                for name in FULL_METT_CONSTRUCTION_FIELDS
            }
        ).to(device)
    else:
        correction = A152METTCorrection(**construction).to(device)
    optimizer = torch.optim.AdamW(
        correction.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs)
    )
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        loader = build_a15_2_epoch_loader(
            dataset,
            epoch=epoch,
            workers=workers,
            cuda=device.type == "cuda",
            sample_order_seed=int(sample_order_seed),
        )
        metrics = run_a15_2_correction_epoch(
            anchor,
            correction,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            anchor_state_reference=anchor_initial_state,
        )
        metrics.pop("_last_replay_batch", None)
        # Per-step diagnostics are useful during debugging but add no metric
        # information to the experimental artifact.
        metrics.pop("step_trace", None)
        source_order = metrics.pop("source_index_order", None)
        _require(
            isinstance(source_order, list)
            and len(source_order) == len(samples)
            and len(set(source_order)) == len(samples),
            "METT epoch physical sample coverage differs",
        )
        metrics["physical_sample_order_is_permutation"] = True
        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch + 1,
            "elapsed_seconds": elapsed,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "metrics": metrics,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "epoch": row["epoch"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "learning_rate": row["learning_rate"],
                    "loss": metrics["loss_components_active_row_weighted"][
                        "total"
                    ],
                    "active_fraction": metrics["active_fraction"],
                    "all_semantic_gradients_nonzero": all(
                        metrics["gradient_nonzero_seen"].values()
                    ),
                    "posterior_integrity": metrics[
                        "posterior_mass_cdf_every_step"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        scheduler.step()

    total_elapsed = time.perf_counter() - started
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture": A15_2_METT_ARCHITECTURE,
        "experiment_variant": {
            "use_tangent": bool(use_tangent),
            "use_shape": bool(use_shape),
            "use_direct_scalar_residual": bool(use_direct_scalar_residual),
        },
        "source_direct_checkpoint": str(Path(direct_checkpoint_path).resolve()),
        "source_anchor": anchor_metadata,
        "construction": construction,
        "epochs": int(epochs),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
        },
        "seeds": {
            "initialization": int(initialization_seed),
            "sample_order": int(sample_order_seed),
        },
        "training_data": {
            "manifest": str(Path(correction_train_manifest_path).resolve()),
            "samples": len(samples),
            "scenes": len({sample.scene_stem for sample in samples}),
            "conditions": [
                "perspective_moderate",
                "perspective_severe",
                "combined_severe",
            ],
        },
        "loss": {
            "name": "A15.1 objective with endpoint-tangent no-harm reference",
            "reference_semantics": (
                "geometric_base compatibility key is the reached SARN endpoint"
            ),
        },
        "parameter_counts": {
            "anchor": sum(parameter.numel() for parameter in anchor.parameters()),
            **fteb_parameter_counts(correction),
        },
        "history": history,
        "training_elapsed_seconds": total_elapsed,
        "anchor_state": _state_cpu(anchor),
        "correction_state": _state_cpu(correction),
    }
    if bool(use_direct_scalar_residual):
        checkpoint["publication_model"] = publication_model_identity(
            "direct_scalar"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "epochs": int(epochs),
        "samples": len(samples),
        "training_elapsed_seconds": total_elapsed,
        "terminal_loss": history[-1]["metrics"][
            "loss_components_active_row_weighted"
        ]["total"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correction-train-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--direct-checkpoint", type=Path, default=DEFAULT_DIRECT_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=TERMINAL_EPOCHS)
    parser.add_argument(
        "--initialization-seed", type=int, default=INITIALIZATION_SEED
    )
    parser.add_argument(
        "--sample-order-seed", type=int, default=SAMPLE_ORDER_SEED
    )
    parser.add_argument("--disable-tangent", action="store_true")
    parser.add_argument("--disable-shape", action="store_true")
    parser.add_argument(
        "--remst",
        "--direct-scalar-residual",
        dest="direct_scalar_residual",
        action="store_true",
        help=(
            "train the publication ReMST mechanism; "
            "--direct-scalar-residual is retained as a legacy alias"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_mett(
        correction_train_manifest_path=args.correction_train_manifest,
        direct_checkpoint_path=args.direct_checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        epochs=args.epochs,
        initialization_seed=args.initialization_seed,
        sample_order_seed=args.sample_order_seed,
        use_tangent=(
            not args.disable_tangent and not args.direct_scalar_residual
        ),
        use_shape=(not args.disable_shape and not args.direct_scalar_residual),
        use_direct_scalar_residual=args.direct_scalar_residual,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DIRECT_CHECKPOINT",
    "DIRECT_SCALAR_METT_VARIANT",
    "FULL_METT_VARIANT",
    "METT_VARIANTS",
    "PROTOCOL",
    "load_mett_models",
    "publication_model_identity",
    "train_mett",
    "validate_full_mett_metadata",
    "validate_mett_variant_metadata",
]
