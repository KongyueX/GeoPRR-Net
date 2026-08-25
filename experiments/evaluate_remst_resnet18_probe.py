"""Evaluate ReMST-ResNet18 on the correction-development split only."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.data import Subset

from experiments.a10_pccot_protocol import (
    DEFAULT_SEED as EVALUATION_SEED,
    condition_evaluation_specs,
)
from experiments.a13_correction_dev_protocol import (
    EVALUATION_CONDITIONS,
    EVALUATION_PIXEL_EPOCH,
    EVALUATION_PIXEL_TOTAL_EPOCHS,
    PROJECTIVE_CONDITIONS,
)
from experiments.a15_fteb import frozen_twin_endpoint_forward
from experiments.evaluate_a15_2_fteb_correction_dev import (
    EVALUATION_BATCH_SIZE,
    build_a15_2_dev_evaluation_dataset,
)
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_DEV_MANIFEST,
    load_a13_correction_dev_manifest,
)
from experiments.train_remst_resnet18_probe import (
    DEFAULT_OUTPUT as DEFAULT_CHECKPOINT,
    load_remst_resnet18_probe,
)
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _loader,
)


PROTOCOL: Final[str] = "remst_resnet18_correction_dev_probe_evaluation_v1"
METHOD_Q0: Final[str] = "raw_anchor"
METHOD_QS: Final[str] = "sarn_endpoint"
METHOD_REMST: Final[str] = "remst_resnet18"
METHODS: Final[tuple[str, ...]] = (METHOD_Q0, METHOD_QS, METHOD_REMST)
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_single_backbone_probe/"
    "seed_20262020/dev_results.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _configure(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _method_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "metric rows are empty")
    result: dict[str, Any] = {}
    for method in METHODS:
        errors = [float(row["absolute_error"][method]) for row in rows]
        tail_count = max(1, (len(errors) + 3) // 4)
        result[method] = {
            "samples": len(errors),
            "nmae": sum(errors) / len(errors),
            "cvar25": sum(sorted(errors, reverse=True)[:tail_count])
            / tail_count,
        }
    return result


def _evaluate_remst_loader(
    anchor: torch.nn.Module,
    correction: torch.nn.Module,
    loader: Any,
    *,
    device: torch.device,
    expected_sample_ids: Sequence[str],
    expected_condition: str,
) -> dict[str, Any]:
    anchor.eval()
    correction.eval()
    rows: list[dict[str, Any]] = []
    unavailable_rows = 0
    unavailable_exact_rows = 0
    integrity_all_valid = True
    with torch.inference_mode():
        for raw_batch in loader:
            original = raw_batch["original_view"].to(device)
            sarn = raw_batch["sarn_view"].to(device)
            support = raw_batch["sarn_support_mask"].to(device)
            dataset_active = raw_batch["sarn_active"].to(device).bool()
            homography = raw_batch["raw_to_sarn_homography"].to(device)
            target = raw_batch["target"].to(device).float()
            ids = tuple(str(value) for value in raw_batch["sample_id"])
            scenes = tuple(str(value) for value in raw_batch["scene_stem"])
            names = tuple(str(value) for value in raw_batch["condition_name"])
            _require(
                all(name == expected_condition for name in names),
                "evaluation condition batch differs",
            )
            clean = torch.tensor(
                [name == "clean" for name in names],
                device=device,
                dtype=torch.bool,
            )
            effective_active = dataset_active & ~clean
            endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
            output = forward_a15_correction(
                correction,
                {
                    "raw_posterior": endpoints["raw_posterior"],
                    "sarn_posterior": endpoints["sarn_posterior"],
                    "raw_mean": endpoints["raw_mean"],
                    "sarn_mean": endpoints["sarn_mean"],
                    "raw_features": endpoints["raw_features"],
                    "sarn_features": endpoints["sarn_features"],
                    "sarn_support_mask": support,
                    "sarn_active": effective_active,
                    "raw_to_sarn_homography": homography,
                },
                endpoint_null=False,
            )
            relation = output["relation_available"].bool()
            unavailable = ~relation
            exact = torch.ones_like(relation)
            if bool(unavailable.any()):
                exact[unavailable] = (
                    (
                        output["progress_posterior"][unavailable]
                        == endpoints["raw_posterior"][unavailable]
                    ).all(dim=1)
                    & (
                        output["mean"][unavailable]
                        == endpoints["raw_mean"][unavailable]
                    )
                )
            unavailable_rows += int(unavailable.sum().cpu())
            unavailable_exact_rows += int((unavailable & exact).sum().cpu())
            posterior = output["progress_posterior"].float()
            cdf = output["progress_cdf"].float()
            integrity = (
                bool(torch.isfinite(posterior).all())
                and bool((posterior >= 0.0).all())
                and bool(
                    torch.allclose(
                        posterior.sum(dim=1),
                        torch.ones_like(target),
                        atol=2.0e-5,
                        rtol=0.0,
                    )
                )
                and bool((cdf[:, 1:] >= cdf[:, :-1]).all())
                and bool(
                    torch.allclose(
                        cdf[:, -1],
                        torch.ones_like(target),
                        atol=2.0e-5,
                        rtol=0.0,
                    )
                )
            )
            integrity_all_valid &= integrity
            _require(integrity, "ReMST posterior mass/CDF integrity differs")
            means = {
                METHOD_Q0: output["raw_anchor_mean"].float(),
                METHOD_QS: output["sarn_endpoint_mean"].float(),
                METHOD_REMST: output["mean"].float(),
            }
            for index, sample_id in enumerate(ids):
                truth = float(target[index])
                predictions = {
                    method: float(value[index]) for method, value in means.items()
                }
                rows.append(
                    {
                        "sample_id": sample_id,
                        "scene_stem": scenes[index],
                        "condition": names[index],
                        "normalized_target": truth,
                        "prediction": predictions,
                        "absolute_error": {
                            method: abs(value - truth)
                            for method, value in predictions.items()
                        },
                        "relation_available": bool(relation[index]),
                    }
                )
    _require(
        tuple(row["sample_id"] for row in rows)
        == tuple(str(value) for value in expected_sample_ids),
        "evaluation sample roster/order differs",
    )
    return {
        "rows": len(rows),
        "relation_available_rows": sum(
            bool(row["relation_available"]) for row in rows
        ),
        "relation_unavailable_rows": unavailable_rows,
        "relation_unavailable_exact_q0_rows": unavailable_exact_rows,
        "posterior_mass_cdf_all_valid": integrity_all_valid,
        "metrics": {"methods": _method_metrics(rows)},
        "per_sample": rows,
    }


def _pool_condition_results(
    conditions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def pool(names: Sequence[str]) -> dict[str, Any]:
        rows = [
            row
            for name in names
            for row in conditions[name]["per_sample"]
        ]
        return {
            "conditions": list(names),
            "rows": len(rows),
            "methods": _method_metrics(rows),
        }

    return {
        "all_conditions": pool(EVALUATION_CONDITIONS),
        "projective_conditions": pool(PROJECTIVE_CONDITIONS),
    }


def _condition_digest(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(value["rows"]),
        "relation_available_rows": int(value["relation_available_rows"]),
        "relation_unavailable_rows": int(value["relation_unavailable_rows"]),
        "relation_unavailable_exact_q0_rows": int(
            value["relation_unavailable_exact_q0_rows"]
        ),
        "posterior_mass_cdf_all_valid": bool(
            value["posterior_mass_cdf_all_valid"]
        ),
        "metrics": value["metrics"],
    }


def evaluate_remst_resnet18_probe(
    *,
    checkpoint_path: Path,
    dev_manifest_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = EVALUATION_BATCH_SIZE,
    max_physical_samples: int | None = None,
) -> dict[str, Any]:
    _require(workers >= 0, "workers must be non-negative")
    _require(batch_size >= 1, "batch size must be positive")
    _require(
        max_physical_samples is None or max_physical_samples >= 1,
        "max physical samples must be positive when supplied",
    )
    output = Path(output_path).resolve()
    _require(not output.exists(), f"evaluation output already exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure(EVALUATION_SEED, device)

    anchor, correction, checkpoint_metadata = load_remst_resnet18_probe(
        checkpoint_path, device=device
    )
    all_samples = tuple(
        load_a13_correction_dev_manifest(Path(dev_manifest_path).resolve())
    )
    samples = (
        all_samples
        if max_physical_samples is None
        else all_samples[: int(max_physical_samples)]
    )
    _require(bool(samples), "correction-dev sample selection is empty")
    expected_ids = tuple(sample.sample_id for sample in samples)
    specs = condition_evaluation_specs(EVALUATION_SEED)
    _require(
        tuple(spec.condition for spec in specs) == EVALUATION_CONDITIONS,
        "fixed correction-dev condition roster differs",
    )
    conditions: dict[str, dict[str, Any]] = {}
    for spec in specs:
        full_dataset = build_a15_2_dev_evaluation_dataset(
            all_samples,
            seed=spec.dataset_seed,
            total_epochs=spec.dataset_total_epochs,
            condition=spec.condition,
        )
        full_dataset.set_epoch(spec.transform_epoch)
        dataset = (
            full_dataset
            if max_physical_samples is None
            else Subset(full_dataset, range(len(samples)))
        )
        loader = _loader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            workers=int(workers),
            seed=spec.loader_seed,
            cuda=device.type == "cuda",
        )
        conditions[spec.condition] = _evaluate_remst_loader(
            anchor,
            correction,
            loader,
            device=device,
            expected_sample_ids=expected_ids,
            expected_condition=spec.condition,
        )
        digest = _condition_digest(conditions[spec.condition])
        print(
            json.dumps(
                {
                    "condition": spec.condition,
                    "rows": digest["rows"],
                    "q0_nmae": digest["metrics"]["methods"][METHOD_Q0][
                        "nmae"
                    ],
                    "q_sarn_nmae": digest["metrics"]["methods"][METHOD_QS][
                        "nmae"
                    ],
                    "remst_resnet18_nmae": digest["metrics"]["methods"][
                        METHOD_REMST
                    ]["nmae"],
                    "relation_available_rows": digest[
                        "relation_available_rows"
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    pooled = _pool_condition_results(conditions)
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "correction_development_only": True,
            "formal_holdout_access": False,
            "field_photo_access": False,
            "automatic_model_selection": False,
            "partial_sample_prefix": max_physical_samples is not None,
        },
        "checkpoint": checkpoint_metadata,
        "data": {
            "manifest": str(Path(dev_manifest_path).resolve()),
            "available_physical_samples": len(all_samples),
            "evaluated_physical_samples": len(samples),
            "conditions": list(EVALUATION_CONDITIONS),
            "projective_conditions": list(PROJECTIVE_CONDITIONS),
            "pixel_total_epochs": EVALUATION_PIXEL_TOTAL_EPOCHS,
            "transform_epoch": EVALUATION_PIXEL_EPOCH,
        },
        "methods": list(METHODS),
        "conditions": {
            name: _condition_digest(value) for name, value in conditions.items()
        },
        "pooled": pooled,
        "evidence": {
            "raw_and_sarn_same_anchor_parameter_set": True,
            "raw_and_sarn_anchor_calls_separate": True,
            "clean_exact_q0": int(conditions["clean"]["relation_available_rows"])
            == 0,
            "all_unavailable_rows_exact_q0": all(
                int(value["relation_unavailable_exact_q0_rows"])
                == int(value["relation_unavailable_rows"])
                for value in conditions.values()
            ),
            "posterior_mass_cdf_all_valid": all(
                bool(value["posterior_mass_cdf_all_valid"])
                for value in conditions.values()
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "complete",
        "output": str(output),
        "evaluated_physical_samples": len(samples),
        "clean_nmae": conditions["clean"]["metrics"]["methods"][
            METHOD_REMST
        ]["nmae"],
        "projective_pooled_nmae": pooled["projective_conditions"]["methods"][
            METHOD_REMST
        ]["nmae"],
        "all_conditions_nmae": pooled["all_conditions"]["methods"][
            METHOD_REMST
        ]["nmae"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dev-manifest", type=Path, default=DEFAULT_CORRECTION_DEV_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=EVALUATION_BATCH_SIZE)
    parser.add_argument("--max-physical-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_remst_resnet18_probe(
        checkpoint_path=args.checkpoint,
        dev_manifest_path=args.dev_manifest,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        max_physical_samples=args.max_physical_samples,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "PROTOCOL",
    "evaluate_remst_resnet18_probe",
]
