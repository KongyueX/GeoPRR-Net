"""One frozen best60 source-validation pass and fixed state interventions."""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.geoprr_v2_data import normalized_target
from experiments.geoprr_v2_relative_state import RelativeStateReader, relative_targets
from experiments.run_geoprr_v2_optimization import write_json
from experiments.summarize_geoprr_v2_relative_state import compare_to_ordinary, read_predictions
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, make_loader, seed_everything
from experiments.train_geoprr_v2_relative_state import (
    PROTOCOL, RelativeStateDataset, save_predictions, validation_rows, validation_summary,
)


READOUTS = ("joint", "product_marginals", "gt_delta_pred_span", "pred_delta_gt_span")


def target_pmfs(delta, span, bins):
    dl, du, dw, ll, lu, lw = relative_targets(delta, span, bins)
    a = delta.new_zeros((len(delta), bins), dtype=torch.float32)
    b = a.clone()
    for p, lower, upper, weight in ((a, dl, du, dw), (b, ll, lu, lw)):
        p.scatter_add_(1, lower[:, None], (1 - weight)[:, None])
        p.scatter_add_(1, upper[:, None], weight[:, None])
    return a, b


def intervention_joints(joint, delta, span):
    """Use unconditional marginals of the original Q; never reweight components."""
    pd, pl = joint.sum(2), joint.sum(1)
    gd, gl = target_pmfs(delta, span, joint.shape[-1])
    return {"joint": joint, "product_marginals": pd[:, :, None] * pl[:, None, :],
            "gt_delta_pred_span": gd[:, :, None] * pl[:, None, :],
            "pred_delta_gt_span": pd[:, :, None] * gl[:, None, :]}


def angular_metrics(probability, truth, *, circular):
    """All angle errors are degrees; entropy is nats, resultant is dimensionless."""
    p = probability.float()
    bins = p.shape[-1]
    x = torch.arange(bins, device=p.device) * (2 * math.pi / bins) if circular else torch.linspace(0, 2 * math.pi, bins, device=p.device)
    residual = lambda a: torch.remainder(a + math.pi, 2 * math.pi) - math.pi if circular else a
    if circular:
        sine, cosine = (p * x.sin()).sum(1), (p * x.cos()).sum(1)
        mean = torch.atan2(sine, cosine).remainder(2 * math.pi)
        resultant = torch.hypot(sine, cosine)
    else:
        mean = (p * x).sum(1)
    peak = x[p.argmax(1)]
    position = truth.remainder(2 * math.pi) * bins / (2 * math.pi) if circular else truth.clamp(0, 2 * math.pi) * (bins - 1) / (2 * math.pi)
    lower = position.floor().long()
    upper = (lower + 1) % bins if circular else (lower + 1).clamp(max=bins - 1)
    adjacent = p.gather(1, lower[:, None]).squeeze(1)
    adjacent += p.gather(1, upper[:, None]).squeeze(1) * (upper != lower)
    result = {"mean_degrees": mean.rad2deg(), "peak_degrees": peak.rad2deg(),
            "mean_error_degrees": residual(mean - truth).abs().rad2deg(),
            "peak_error_degrees": residual(peak - truth).abs().rad2deg(),
            "expected_error_degrees": (p * residual(x[None, :] - truth[:, None]).abs()).sum(1).rad2deg(),
            "adjacent_mass": adjacent, "entropy_nats": -(p * p.clamp_min(1e-30).log()).sum(1)}
    if circular:
        result["resultant_length"] = resultant
    return result


def component_validity(output):
    valid = torch.ones(len(output["mixture_probs"]), dtype=torch.bool, device=output["joint"].device)
    for key in ("mixture_probs", "delta_probs", "span_probs"):
        p = output[key].float()
        valid &= torch.isfinite(p).flatten(1).all(1) & (p >= 0).flatten(1).all(1)
        valid &= ((p.sum(-1) - 1).abs() <= 1e-4).reshape(len(p), -1).all(1)
    return valid


def diagnostic_batch(output, batch):
    q = output["joint"]
    values = {}
    for role, p, truth, circular in (("delta", q.sum(2), batch["delta_radians"], True),
                                      ("span", q.sum(1), batch["span_radians"], False)):
        for key, value in angular_metrics(p, truth, circular=circular).items():
            values[f"{role}_{key}"] = value
        for component in range(output["mixture_probs"].shape[1]):
            cp = output[f"{role}_probs"][:, component]
            for key, value in angular_metrics(cp, truth, circular=circular).items():
                values[f"{role}_component{component}_{key}"] = value
        a, b = values[f"{role}_component0_mean_degrees"], values[f"{role}_component1_mean_degrees"]
        values[f"{role}_component_mean_separation_degrees"] = ((a - b + 180) % 360 - 180).abs() if circular else (a - b).abs()
    for component in range(output["mixture_probs"].shape[1]):
        values[f"mixture{component}"] = output["mixture_probs"][:, component]
    values["components_valid"] = component_validity(output)
    arrays = {key: value.detach().cpu().tolist() for key, value in values.items()}
    return [{key: (value[i] if isinstance(value[i], bool) or math.isfinite(value[i]) else None)
             for key, value in arrays.items()} for i in range(len(batch["y"]))]


def verify_population(dataset, reference, ordinary, checkpoint):
    expected = {r["sample_id"]: r for r in reference}
    old = {r["sample_id"]: r for r in ordinary}
    if set(expected) != set(old) or set(expected) != {r["sample_id"] for r in dataset.rows}:
        raise ValueError("validation/reference/ordinary source IDs differ")
    if dataset.supervision_contract() != checkpoint["data_contract"]["validation"]:
        raise ValueError("validation geometry/manifest differs from checkpoint")
    for source in dataset.rows:
        sid = source["sample_id"]
        for row in (expected[sid], old[sid]):
            if row["group_id"] != source["group_id"] or float(row["target"]) != float(np.float32(normalized_target(source))):
                raise ValueError(f"{sid}: group or float32 target differs from saved reference")


def state_summary(rows):
    cohorts = {"full1506": rows,
               "common_ordinary1495": [r for r in rows if r["common_ordinary"]],
               "r8_ordinary1497": [r for r in rows if r["r8_ordinary"]],
               "old_severe_union6": [r for r in rows if r["old_severe"]],
               "joint_new_above2_vs_r8": [r for r in rows if r["joint_absolute_error"] > .02 >= r["r8_reference_absolute_error"]]}
    result = {}
    keys = [k for k in rows[0] if k.startswith(("delta_", "span_", "mixture"))]
    for name, selected in cohorts.items():
        stats = {k: [r[k] for r in selected if r[k] is not None] for k in keys}
        result[name] = {"samples": len(selected), "sample_ids": [r["sample_id"] for r in selected],
                        "state_metrics": {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                              "p90": float(np.quantile(v, .9))} for k, v in stats.items() if v}}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "val-manifest", "val-labels", "reference-predictions", "ordinary-reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "diagnosis.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    if any(path != lock for path in args.output.iterdir()):
        lock.unlink()
        raise FileExistsError("diagnosis output is nonempty; retain it and do not repeat the model pass")
    started = time.perf_counter()
    try:
        device = torch.device("cuda")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if (checkpoint["protocol"] != PROTOCOL or checkpoint["epoch"] != 60 or checkpoint["best_epoch"] != 60
                or config["model"] != {"bins": 256, "decoder_dim": 64, "components": 2, "coupling": "joint"}
                or config["training"]["seed"] != 20262020 or config["training"]["batch_size"] != 32
                or config["training"]["workers"] != 8 or config["training"]["amp"] is not True
                or config["data"] != {"image_size": 768, "output_size": 192}):
            raise ValueError("diagnosis requires the frozen joint best60 configuration")
        reference, ordinary = read_predictions(args.reference_predictions), read_predictions(args.ordinary_reference)
        ref_by_id, old_by_id = ({r["sample_id"]: r for r in records} for records in (reference, ordinary))
        data = RelativeStateDataset(args.val_manifest, args.val_labels, data_root=args.data_root,
                                    data_config=config["data"], augment=False)
        verify_population(data, reference, ordinary, checkpoint)
        seed_everything(20262020)
        model = RelativeStateReader(**config["model"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        loader = make_loader(data, batch_size=32, workers=8, shuffle=False, seed=20262020, device=device)
        protocol = {"protocol": "geoprr_v2_relative_state_frozen_best60_diagnosis_v1", "checkpoint_epoch": 60,
                    "code_revision": os.environ.get("GEOPRR_CODE_REVISION"),
                    "config": config, "network_forward_images_planned": 1506, "optimizer_updates": 0,
                    "readouts": list(READOUTS), "prediction_input": "RGB only",
                    "marginals": "sum original joint over the other axis; no extra normalization",
                    "gt_interventions": "unconditional outer product; GT replacement is diagnostic and not deployable",
                    "readout": "unchanged full-mass native mean; endpoint mass uniform; no fallback or r8",
                    "replay_tolerance_normalized_reading": 1e-5,
                    "checkpoint": str(args.checkpoint), "reference_predictions": str(args.reference_predictions),
                    "ordinary_reference": str(args.ordinary_reference), "state_labels": str(args.val_labels)}
        write_json(args.output / "protocol.json", protocol)
        torch.cuda.reset_peak_memory_stats(device)
        predictions = {name: [] for name in READOUTS}
        diagnostics, caches = [], {k: [] for k in ("mixture", "delta_components", "span_components", "gt_delta", "gt_span")}
        forward_seconds = counterfactual_seconds = 0.
        forwarded = 0
        with torch.inference_mode():
            for raw in loader:
                batch = device_batch(raw, device)
                torch.cuda.synchronize(device)
                tick = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    output = model(batch["image"])
                torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter() - tick
                forwarded += len(batch["y"])
                valid = component_validity(output).cpu().tolist()
                checked_batch = {**batch, "input_failure": [failure or ("" if good else "invalid_component_pmf")
                                 for failure, good in zip(batch["input_failure"], valid)]}
                predictions["joint"].extend(validation_rows(output, checked_batch))
                tick = time.perf_counter()
                interventions = intervention_joints(output["joint"], batch["delta_radians"], batch["span_radians"])
                for name in READOUTS[1:]:
                    predictions[name].extend(validation_rows(model.readout(interventions[name]), checked_batch))
                torch.cuda.synchronize(device)
                counterfactual_seconds += time.perf_counter() - tick
                details = diagnostic_batch(output, batch)
                for sid, detail in zip(batch["sample_id"], details):
                    diagnostics.append({"sample_id": sid, "group_id": old_by_id[sid]["group_id"],
                                        "target": float(old_by_id[sid]["target"]), **detail})
                for key, tensor in (("mixture", output["mixture_probs"]), ("delta_components", output["delta_probs"]),
                                    ("span_components", output["span_probs"]), ("gt_delta", batch["delta_radians"]),
                                    ("gt_span", batch["span_radians"])):
                    caches[key].append(tensor.float().cpu().numpy())
        if forwarded != 1506 or {r["sample_id"] for r in diagnostics} != set(ref_by_id):
            raise RuntimeError("one-pass validation population is incomplete")
        replay_differences, replay_failures = [], []
        for index, row in enumerate(diagnostics):
            sid = row["sample_id"]
            original, replay = ref_by_id[sid], predictions["joint"][index]
            old = old_by_id[sid]
            native, modal = float(old["native_reference_absolute_error"]), float(old["r8_reference_absolute_error"])
            saved_valid = original["valid"].lower() == "true"
            difference = abs(replay["prediction"] - float(original["prediction"])) if replay["valid"] and saved_valid else None
            if saved_valid != replay["valid"] or (difference is not None and difference > 1e-5):
                replay_failures.append(sid)
            if difference is not None:
                replay_differences.append(difference)
            row.update({"replay_absolute_difference": difference, "native_reference_absolute_error": native,
                        "r8_reference_absolute_error": modal, "common_ordinary": native <= .01 and modal <= .01,
                        "r8_ordinary": modal <= .01, "old_severe": native > .02 or modal > .02})
            for name in READOUTS:
                predicted = predictions[name][index]
                row.update({f"{name}_{key}": predicted[key] for key in
                            ("prediction", "absolute_error", "valid", "invalid_mass", "outside_arc_mass")})
        np.savez_compressed(args.output / "components.npz", sample_ids=np.array([r["sample_id"] for r in diagnostics]),
                            group_ids=np.array([r["group_id"] for r in diagnostics]),
                            target=np.array([r["target"] for r in diagnostics], dtype=np.float32),
                            **{key: np.concatenate(value) for key, value in caches.items()})
        save_predictions(args.output / "diagnostics.csv.gz", diagnostics)
        comparisons = {}
        for name in READOUTS:
            path = args.output / f"{name}.csv.gz"
            save_predictions(path, predictions[name])
            comparisons[name] = {"validation": validation_summary(predictions[name]),
                                 "ordinary_comparison": compare_to_ordinary(path, args.ordinary_reference)}
        status = "replay_mismatch" if replay_failures else "complete"
        write_json(args.output / "metrics.json", {"status": status, "comparisons": comparisons,
                   "replay": {"maximum_absolute_difference": max(replay_differences, default=None),
                              "mismatch_ids": replay_failures}, "state_cohorts": state_summary(diagnostics),
                   "interpretation_allowed": not replay_failures,
                   "scope": "frozen-checkpoint source diagnosis; no trained coupling-control or deployable GT result"})
        costs = {"network_forward_images": forwarded, "optimizer_updates": 0,
                 "network_and_original_readout_seconds": forward_seconds,
                 "three_counterfactuals_seconds": counterfactual_seconds,
                 "process_wall_seconds": time.perf_counter() - started,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "torch": torch.__version__, "cuda": torch.version.cuda,
                 "gpu": torch.cuda.get_device_name(device),
                 "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                 "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        write_json(args.output / "costs.json", costs)
        write_json(args.output / "completed.json", {"status": status, "samples": forwarded, "seed": 20262020,
                   "checkpoint_epoch": 60, "optimizer_updates": 0, "interpretation_allowed": not replay_failures})
        print({"status": status, "samples": forwarded, "replay_mismatches": len(replay_failures)}, flush=True)
        return 2 if replay_failures else 0
    except Exception as exc:
        write_json(args.output / "failed.json", {"exception": type(exc).__name__, "message": str(exc),
                   "process_wall_seconds": time.perf_counter() - started, "partial_retained": True})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
