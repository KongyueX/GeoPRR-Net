"""Train the frozen-protocol single-seed PG-SIAM-v3 screen candidate.

Only the two diagonal layer-3/layer-4 fusion vectors are optimized.  The
frozen DB-GAR18 parent, BatchNorm statistics, progress head, and geometry head
remain unchanged.  Training uses SyncG fit rows and only the eight XM2
development fit groups declared by ``pgsiam_v3_protocol.json``; the three
selection groups and all external/holdout cohorts are rejected from training.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from experiments import perspective_adaptive_db_gar18 as paired
from experiments.domain_balanced_geoattn_resnet18 import (
    ARCHITECTURE as PARENT_ARCHITECTURE,
    EXPECTED_REAL_DEVELOPMENT_SAMPLES,
    EXPECTED_REAL_GROUPS,
    EXPECTED_SYNTHETIC_FIT_SAMPLES,
    EXPECTED_SYNTHETIC_HOLDOUT_IDS,
    PROTOCOL as PARENT_PROTOCOL,
    REAL_AUGMENT_ANCHORS,
    RealProgressSample,
    _sha256_file,
    load_real_progress_samples,
    strong_real_augmentation,
)
from experiments.geoattn_resnet18_progress import (
    DIRECTION_LOSS_WEIGHT,
    PIVOT_LOSS_WEIGHT,
    REFERENCE_LOSS_WEIGHT,
    _apply_homography,
    _geometry_targets,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.projective_geometry_guided_siam import (
    ARCHITECTURE,
    METHOD_PREFIX,
    PROTOCOL,
    TRAINABLE_PARAMETERS,
    ProjectiveGeometryGuidedSIAM,
    load_db_gar_state_into_pg_siam_model,
    set_pg_siam_calibration_stage,
)
from experiments.projective_geometry_views import (
    build_projective_geometry_views,
    resize_projective_geometry_views,
)
from experiments.resnet18_direct_progress import (
    DEFAULT_MANIFEST,
    IMAGE_SIZE,
    DirectSample,
    _canonical_json_bytes,
    _canonical_sha256,
    _configure_reproducibility,
    load_training_samples,
    matched_cagh_augmentation,
)
from experiments.run_cagh_v5_plain_paper_batch import load_canonical_roi
from experiments.sarn_guided_attention_db_gar18 import (
    _transform_points_after_sarn_crop,
)
from experiments.train_cagh_scalemark_reference_probe_v5 import (
    _augment_geometry as _cagh_augment_geometry,
    _augment_photo as _cagh_augment_photo,
)
from experiments.v5_shared_roi_comparison_input import (
    canonical_tight_roi_native,
    direct_resize_whole_roi,
)


SCREEN_PROTOCOL: Final[str] = "projective_geometry_guided_siam_v3_single_seed_screen_v1"
PROTOCOL_PATH: Final[Path] = Path(__file__).with_name("pgsiam_v3_protocol.json")
EPOCHS: Final[int] = 2
BATCH_SIZE: Final[int] = 16
LEARNING_RATE: Final[float] = 3e-4
WEIGHT_DECAY: Final[float] = 1e-4
REAL_FRACTION: Final[float] = 0.25
CVaR_TAIL_FRACTION: Final[float] = 0.10
CVaR_WEIGHT: Final[float] = 1.0
CONSISTENCY_WEIGHT: Final[float] = 0.25
GEOMETRY_WEIGHT: Final[float] = 0.25
HUBER_BETA: Final[float] = 0.05
REAL_SEED_OFFSET: Final[int] = 8_120_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_frozen_protocol(path: Path = PROTOCOL_PATH) -> Mapping[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), "v3 protocol is not an object")
    _require(value.get("protocol") == SCREEN_PROTOCOL, "v3 screen protocol mismatch")
    training = value.get("training")
    _require(isinstance(training, Mapping), "v3 training protocol is missing")
    _require(training.get("epochs") == EPOCHS, "v3 epoch protocol drift")
    _require(training.get("target_bin_balancing") is False, "target balancing must be off")
    _require(
        training.get("new_trainable_parameters") == TRAINABLE_PARAMETERS,
        "v3 parameter protocol drift",
    )
    return value


def split_real_fit_selection(
    samples: Sequence[RealProgressSample], protocol: Mapping[str, Any]
) -> tuple[tuple[RealProgressSample, ...], tuple[RealProgressSample, ...]]:
    training = protocol["training"]
    selection_groups = frozenset(str(value) for value in training["real_selection_groups"])
    fit = tuple(sample for sample in samples if sample.group_id not in selection_groups)
    selection = tuple(sample for sample in samples if sample.group_id in selection_groups)
    _require(
        len(fit) == int(training["real_fit_samples"])
        and len({sample.group_id for sample in fit}) == int(training["real_fit_groups"]),
        "real fit roster does not match frozen protocol",
    )
    _require(
        len(selection) == int(training["real_selection_samples"])
        and {sample.group_id for sample in selection} == selection_groups,
        "real selection roster does not match frozen protocol",
    )
    _require(
        not ({sample.sample_id for sample in fit} & {sample.sample_id for sample in selection}),
        "real fit/selection samples overlap",
    )
    return fit, selection


class DomainMixtureSampler(Sampler[tuple[int, int]]):
    """One synthetic-length schedule with label-agnostic real replacement."""

    def __init__(self, synthetic_count: int, real_count: int, *, draws: int, seed: int) -> None:
        _require(synthetic_count > 0 and real_count > 0 and draws > 0, "empty sampler")
        self.synthetic_count = int(synthetic_count)
        self.real_count = int(real_count)
        self.draws = int(draws)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(self.synthetic_count).tolist()
        cursor = 0
        for draw in range(self.draws):
            if bool(rng.random() < REAL_FRACTION):
                yield self.synthetic_count + int(rng.integers(0, self.real_count)), draw
            else:
                if cursor >= len(order):
                    order = rng.permutation(self.synthetic_count).tolist()
                    cursor = 0
                yield int(order[cursor]), draw
                cursor += 1

    def __len__(self) -> int:
        return self.draws


class PGSIAMTrainingDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        synthetic: Sequence[DirectSample],
        real: Sequence[RealProgressSample],
        *,
        seed: int,
    ) -> None:
        self.synthetic = tuple(synthetic)
        self.real = tuple(real)
        self.seed = int(seed)
        self.epoch = 0
        self.synthetic_augmentation = matched_cagh_augmentation()
        self.real_augmentation = strong_real_augmentation()
        self._real_images = tuple(self._load_real(index) for index in range(len(self.real)))

    def __len__(self) -> int:
        return len(self.synthetic) + len(self.real)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_real(self, index: int) -> np.ndarray:
        _payload, image = load_canonical_roi(self.real[index].roi)
        return direct_resize_whole_roi(image, size=IMAGE_SIZE)

    def _synthetic_roi_points(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.synthetic[index]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source decode failed")
        roi, bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        roi = direct_resize_whole_roi(roi, size=IMAGE_SIZE)
        _require(bool(sample.protected_points_xy), f"{sample.sample_id}: geometry missing")
        left, top, right, bottom = bounds
        points = (
            np.asarray(sample.protected_points_xy, dtype=np.float32)
            - np.asarray([left, top], dtype=np.float32)
        ) / np.asarray([right - left, bottom - top], dtype=np.float32)
        return np.ascontiguousarray(roi), points

    def __getitem__(self, raw_index: int | tuple[int, int]) -> dict[str, torch.Tensor]:
        if isinstance(raw_index, tuple):
            source_index, draw = int(raw_index[0]), int(raw_index[1])
        else:
            source_index, draw = int(raw_index), 0
        is_real = source_index >= len(self.synthetic)
        if is_real:
            local = source_index - len(self.synthetic)
            image = self._real_images[local].copy()
            points = None
            target = self.real[local].normalized_target
            augmentation = self.real_augmentation
            anchors = REAL_AUGMENT_ANCHORS
            offset = REAL_SEED_OFFSET
        else:
            local = source_index
            image, points = self._synthetic_roi_points(local)
            target = self.synthetic[local].normalized_target
            augmentation = self.synthetic_augmentation
            anchors = points
            offset = 0
        rng = np.random.default_rng(
            self.seed + offset + self.epoch * 1_000_003 + local * 97 + draw * 7_919
        )
        image, forward, _code = _cagh_augment_geometry(image, anchors, rng, augmentation)
        if points is not None:
            points = _apply_homography(points, forward)
        image, _photo = _cagh_augment_photo(image, rng, augmentation)
        projective_rng = np.random.default_rng(
            self.seed
            + offset
            + paired.PERSPECTIVE_SEED_OFFSET
            + self.epoch * 2_000_033
            + local * 193
            + draw * 15_869
        )
        projective, projective_points, _metadata = paired._paired_projective_view(
            image,
            points,
            rng=projective_rng,
            epoch=self.epoch,
            total_epochs=EPOCHS,
        )
        views = resize_projective_geometry_views(
            build_projective_geometry_views(projective), output_hw=(IMAGE_SIZE, IMAGE_SIZE)
        )
        projective_points = _transform_points_after_sarn_crop(
            projective_points,
            views.sarn_decision,
            height=int(projective.shape[0]),
            width=int(projective.shape[1]),
        )
        row: dict[str, torch.Tensor] = {
            "clean_image": normalized_rgb_tensor(image),
            "view_a": normalized_rgb_tensor(views.view_a_bgr),
            "view_b": normalized_rgb_tensor(views.view_b_bgr),
            "homography_a_to_b": torch.from_numpy(
                views.homography_a_to_b.astype(np.float32, copy=False)
            ),
            "confidence": torch.tensor(views.confidence, dtype=torch.float32),
            "active": torch.tensor(views.active, dtype=torch.bool),
            "progress": torch.tensor(target, dtype=torch.float32),
            "is_real": torch.tensor(is_real, dtype=torch.bool),
        }
        if projective_points is None:
            row.update(
                {
                    "pivot": torch.zeros(2),
                    "direction_sin_cos": torch.zeros(2),
                    "references": torch.zeros(4),
                    "has_geometry": torch.tensor(False),
                }
            )
        else:
            labels = _geometry_targets(projective_points)
            row.update(labels)
            row["has_geometry"] = torch.tensor(True)
        return row


def smooth_l1_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target, beta=HUBER_BETA, reduction="none")


def cvar_no_harm_loss(
    candidate_loss: torch.Tensor,
    parent_loss: torch.Tensor,
    *,
    tail_fraction: float = CVaR_TAIL_FRACTION,
) -> torch.Tensor:
    _require(candidate_loss.shape == parent_loss.shape, "CVaR loss shape mismatch")
    _require(candidate_loss.ndim == 1 and candidate_loss.numel() > 0, "CVaR loss is empty")
    count = max(1, int(math.ceil(float(tail_fraction) * candidate_loss.numel())))
    excess = candidate_loss - parent_loss.detach()
    return torch.relu(torch.topk(excess, k=count, largest=True).values.mean())


def _domain_balanced_mean(values: torch.Tensor, is_real: torch.Tensor) -> torch.Tensor:
    _require(values.ndim == 1 and values.shape == is_real.shape, "domain mean shape mismatch")
    real_mask = is_real.bool()
    synthetic_mask = ~real_mask
    present: list[torch.Tensor] = []
    if bool(torch.any(synthetic_mask)):
        present.append(values[synthetic_mask].mean())
    if bool(torch.any(real_mask)):
        present.append(values[real_mask].mean())
    _require(bool(present), "domain mean is empty")
    return torch.stack(present).mean()


def domain_balanced_cvar_no_harm_loss(
    candidate_loss: torch.Tensor,
    parent_loss: torch.Tensor,
    is_real: torch.Tensor,
) -> torch.Tensor:
    """Average domain-specific tail risks so one domain cannot hide another."""

    _require(candidate_loss.shape == parent_loss.shape == is_real.shape, "domain CVaR shape mismatch")
    real_mask = is_real.bool()
    values: list[torch.Tensor] = []
    for mask in (~real_mask, real_mask):
        if bool(torch.any(mask)):
            values.append(cvar_no_harm_loss(candidate_loss[mask], parent_loss[mask]))
    _require(bool(values), "domain CVaR is empty")
    return torch.stack(values).mean()


def pgsiam_objective(
    outputs: Mapping[str, torch.Tensor],
    parent_clean: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    candidate_losses = smooth_l1_per_sample(outputs["progress"], batch["progress"])
    parent_losses = smooth_l1_per_sample(outputs["parent_progress"], batch["progress"])
    supervised = _domain_balanced_mean(candidate_losses, batch["is_real"])
    safe = domain_balanced_cvar_no_harm_loss(
        candidate_losses, parent_losses, batch["is_real"]
    )
    consistency_losses = smooth_l1_per_sample(
        outputs["progress"], parent_clean["progress"].detach()
    )
    consistency = _domain_balanced_mean(consistency_losses, batch["is_real"])
    geometry_mask = batch["has_geometry"].bool()
    if bool(torch.any(geometry_mask)):
        pivot = F.smooth_l1_loss(
            outputs["pivot"][geometry_mask], batch["pivot"][geometry_mask], beta=HUBER_BETA
        )
        direction = (
            1.0
            - torch.sum(
                outputs["direction_sin_cos"][geometry_mask]
                * batch["direction_sin_cos"][geometry_mask],
                dim=1,
            )
        ).mean()
        references = F.smooth_l1_loss(
            outputs["references"][geometry_mask],
            batch["references"][geometry_mask],
            beta=HUBER_BETA,
        )
        geometry = (
            PIVOT_LOSS_WEIGHT * pivot
            + DIRECTION_LOSS_WEIGHT * direction
            + REFERENCE_LOSS_WEIGHT * references
        )
    else:
        geometry = supervised * 0.0
    total = (
        supervised
        + CVaR_WEIGHT * safe
        + CONSISTENCY_WEIGHT * consistency
        + GEOMETRY_WEIGHT * geometry
    )
    return total, {
        "supervised": supervised,
        "safe_cvar": safe,
        "clean_consistency": consistency,
        "geometry": geometry,
        "mean_excess": (candidate_losses - parent_losses).mean(),
    }


def _load_parent(checkpoint_path: Path) -> tuple[ProjectiveGeometryGuidedSIAM, Mapping[str, Any], tuple[str, ...]]:
    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"parent checkpoint missing: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "parent checkpoint is not an object")
    _require(checkpoint.get("protocol") == PARENT_PROTOCOL, "parent protocol mismatch")
    _require(checkpoint.get("architecture") == PARENT_ARCHITECTURE, "parent architecture mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "parent model state missing")
    model = ProjectiveGeometryGuidedSIAM(imagenet_pretrained=False)
    missing = load_db_gar_state_into_pg_siam_model(model, state)
    return model, checkpoint, missing


def _train_epoch(
    model: ProjectiveGeometryGuidedSIAM,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    totals = {key: 0.0 for key in (
        "loss", "supervised", "safe_cvar", "clean_consistency", "geometry",
        "mean_excess", "shift_abs", "shift_signed", "effective_gate", "active_fraction",
        "real_fraction",
    )}
    steps = 0
    for raw in loader:
        batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in raw.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model.forward_pair_training(
                batch["view_a"], batch["view_b"], batch["homography_a_to_b"],
                batch["confidence"], batch["active"],
            )
            parent_clean = model.forward_parent_training(batch["clean_image"])
            loss, components = pgsiam_objective(outputs, parent_clean, batch)
        _require(bool(torch.isfinite(loss)), "PG-SIAM loss became non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad), 5.0
        )
        scaler.step(optimizer)
        scaler.update()
        shift = outputs["progress"] - outputs["parent_progress"]
        totals["loss"] += float(loss.detach().cpu())
        for key, value in components.items():
            totals[key] += float(value.detach().cpu())
        totals["shift_abs"] += float(shift.detach().abs().mean().cpu())
        totals["shift_signed"] += float(shift.detach().mean().cpu())
        totals["effective_gate"] += float(outputs["effective_gate"].detach().mean().cpu())
        totals["active_fraction"] += float(batch["active"].float().mean().cpu())
        totals["real_fraction"] += float(batch["is_real"].float().mean().cpu())
        steps += 1
    _require(steps > 0, "PG-SIAM epoch is empty")
    return {**{key: value / steps for key, value in totals.items()}, "steps": float(steps)}


def train(
    *,
    parent_checkpoint_path: Path,
    synthetic_manifest_path: Path,
    synthetic_split_path: Path,
    real_manifest_path: Path,
    real_labels_path: Path,
    output_path: Path,
    seed: int,
    device_name: str = "cuda:0",
    workers: int = 4,
    frozen_protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    protocol_source = Path(frozen_protocol_path).resolve()
    protocol = load_frozen_protocol(protocol_source)
    _require(seed == int(protocol["seed"]), "seed differs from frozen protocol")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"refusing to overwrite v3 checkpoint: {output}")
    synthetic, roster = load_training_samples(synthetic_manifest_path, synthetic_split_path)
    real_all = load_real_progress_samples(real_manifest_path, real_labels_path)
    _require(
        len(synthetic) == EXPECTED_SYNTHETIC_FIT_SAMPLES
        and len(roster.validation_ids) == EXPECTED_SYNTHETIC_HOLDOUT_IDS,
        "SyncG fit/holdout roster drift",
    )
    _require(
        len(real_all) == EXPECTED_REAL_DEVELOPMENT_SAMPLES
        and len({sample.group_id for sample in real_all}) == EXPECTED_REAL_GROUPS,
        "XM2 development roster drift",
    )
    real_fit, real_selection = split_real_fit_selection(real_all, protocol)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA unavailable")
    _configure_reproducibility(seed, device)
    model, parent, missing = _load_parent(parent_checkpoint_path)
    _require(int(parent.get("seed", -1)) == seed, "parent seed mismatch")
    model = model.to(device)
    trainable_names = set_pg_siam_calibration_stage(model)
    dataset = PGSIAMTrainingDataset(synthetic, real_fit, seed=seed)
    draws = int(math.ceil(len(synthetic) / BATCH_SIZE) * BATCH_SIZE)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=DomainMixtureSampler(
                len(synthetic), len(real_fit), draws=draws, seed=seed + epoch
            ),
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        metrics = _train_epoch(
            model, loader, device=device, optimizer=optimizer, scaler=scaler
        )
        row = {"epoch": epoch + 1, "train": metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    source_paths = {
        "training": Path(__file__).resolve(),
        "architecture": Path(__import__("experiments.projective_geometry_guided_siam", fromlist=["x"]).__file__).resolve(),
        "views": Path(__import__("experiments.projective_geometry_views", fromlist=["x"]).__file__).resolve(),
        "rectification": Path(__import__("experiments.support_quad_rectification", fromlist=["x"]).__file__).resolve(),
        "sarn_v2": Path(__import__("experiments.support_aware_roi_normalization_v2", fromlist=["x"]).__file__).resolve(),
        "screen_protocol": protocol_source,
    }
    method = f"{METHOD_PREFIX}_seed_{seed}"
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "screen_protocol": SCREEN_PROTOCOL,
        "architecture": ARCHITECTURE,
        "method": method,
        "seed": seed,
        "image_size": IMAGE_SIZE,
        "screen_scope": "single_seed_internal_selection_only",
        "parent_checkpoint": str(Path(parent_checkpoint_path).resolve()),
        "parent_checkpoint_sha256": _sha256_file(Path(parent_checkpoint_path).resolve()),
        "parent_protocol": parent.get("protocol"),
        "parent_architecture": parent.get("architecture"),
        "parent_missing_keys_initialized": list(missing),
        "source_files": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in source_paths.items()
        },
        "training_inputs": {
            "synthetic_manifest": str(Path(synthetic_manifest_path).resolve()),
            "synthetic_manifest_sha256": _sha256_file(Path(synthetic_manifest_path).resolve()),
            "synthetic_split": str(Path(synthetic_split_path).resolve()),
            "synthetic_split_sha256": _sha256_file(Path(synthetic_split_path).resolve()),
            "synthetic_fit_samples": len(synthetic),
            "synthetic_holdout_samples": len(roster.validation_ids),
            "synthetic_fit_ids_sha256": _canonical_sha256(sorted(roster.train_ids)),
            "synthetic_holdout_ids_sha256": _canonical_sha256(sorted(roster.validation_ids)),
            "real_manifest": str(Path(real_manifest_path).resolve()),
            "real_manifest_sha256": _sha256_file(Path(real_manifest_path).resolve()),
            "real_labels": str(Path(real_labels_path).resolve()),
            "real_labels_sha256": _sha256_file(Path(real_labels_path).resolve()),
            "real_fit_samples": len(real_fit),
            "real_fit_groups": sorted({sample.group_id for sample in real_fit}),
            "real_selection_samples_excluded": len(real_selection),
            "real_selection_groups_excluded": sorted({sample.group_id for sample in real_selection}),
        },
        "schedule": {
            "epochs": EPOCHS,
            "checkpoint_selection": "terminal_fixed_epoch",
            "batch_size": BATCH_SIZE,
            "real_fraction": REAL_FRACTION,
            "target_balancing": False,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "trainable_parameter_names": list(trainable_names),
        },
        "loss": {
            "supervised_progress": True,
            "synthetic_geometry_auxiliary": True,
            "clean_parent_consistency": True,
            "parent_relative_cvar_no_harm": True,
            "cvar_tail_fraction": CVaR_TAIL_FRACTION,
            "domain_balanced_supervision_and_cvar": True,
        },
        "parameter_inventory": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        "training_seconds": float(time.perf_counter() - started),
        "history": history,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    _require(checkpoint["parameter_inventory"]["trainable"] == TRAINABLE_PARAMETERS, "trainable count drift")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "checkpoint": str(output),
        "method": method,
        "training_seconds": checkpoint["training_seconds"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--synthetic-split", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--real-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frozen-protocol", type=Path, default=PROTOCOL_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train(
        parent_checkpoint_path=args.parent_checkpoint,
        synthetic_manifest_path=args.synthetic_manifest,
        synthetic_split_path=args.synthetic_split,
        real_manifest_path=args.real_manifest,
        real_labels_path=args.real_labels,
        output_path=args.output,
        seed=args.seed,
        device_name=args.device,
        workers=args.workers,
        frozen_protocol_path=args.frozen_protocol,
    )
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
