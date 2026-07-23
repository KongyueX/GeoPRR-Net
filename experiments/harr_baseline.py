"""Audited adapter for the released HARR pointer-segmentation branch.

HARR is kept as an ignored external checkout.  This module verifies the
upstream source and checkpoint identities, loads the complete released
checkpoint strictly, and then exposes only the pointer branch required by the
Pointer-10K direction-component protocol.

The adapter deliberately does not call HARR's OCR/reading post-processing.
That pipeline needs dial/text masks and transcripts that SyncG does not
provide, and its public inference script targets OpenCV 3.  The direction
decoder below reproduces the released pointer-mask threshold, skeletonization,
probabilistic Hough transform, and centre-to-tip endpoint convention.
"""
from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from torch import nn

from experiments.vdn_baseline import affine_for_dial


PROJECT_DIR = Path(__file__).resolve().parents[1]
HARR_REPOSITORY = "https://github.com/shuyansy/Detect-and-read-meters.git"
HARR_PINNED_COMMIT = "e5e16803de2c06b3dfb248df16ee91c05879cd61"
HARR_RELEASED_CHECKPOINT_URL = (
    "https://drive.google.com/file/d/"
    "1sHmEEf9E0_kvL0LW1S5Y5jjFgjx_O5Dj/view"
)
HARR_RELEASED_CHECKPOINT_SHA256 = (
    "6f5bcfd5f57c535dbc4da827ba7538e1c305f33d3da84d43215b125500a4300a"
)
HARR_POINTER_PROTOCOL = "harr_v2_released_pointer_branch_zero_shot_v1"
HARR_IMAGE_SIZE = 512
HARR_CROP_EXPANSION = 1.25
HARR_POINTER_THRESHOLD = 0.5


@dataclass(frozen=True)
class HARRDirectionPrediction:
    direction: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor
    failure_reason: tuple[str | None, ...]


class HARRPointerBranch(nn.Module):
    """The active FPN and 1x1 segmentation head from released HARR."""

    def __init__(self, fpn: nn.Module, predict: nn.Module) -> None:
        super().__init__()
        self.fpn = fpn
        self.predict = predict

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.fpn(images)[0]
        return self.predict(features)[:, :1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_harr_source(path: Path) -> str:
    path = path.resolve()
    required = (
        path / "network" / "textnet.py",
        path / "network" / "vgg.py",
        path / "network" / "crnn.py",
        path / "LICENSE.md",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"HARR source is incomplete at {path}: {', '.join(missing)}"
        )
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if commit != HARR_PINNED_COMMIT:
        raise ValueError(
            f"HARR source commit is {commit}, expected {HARR_PINNED_COMMIT}"
        )
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("formal HARR source checkout contains tracked modifications")
    license_text = (path / "LICENSE.md").read_text(
        encoding="utf-8",
        errors="replace",
    )
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise ValueError("HARR checkout does not contain the expected MIT notice")
    return commit


def verify_harr_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
) -> dict[str, torch.Tensor]:
    digest = sha256_file(path)
    if digest != HARR_RELEASED_CHECKPOINT_SHA256:
        raise ValueError(
            f"HARR checkpoint SHA-256 is {digest}, "
            f"expected {HARR_RELEASED_CHECKPOINT_SHA256}"
        )
    if int(checkpoint.get("epoch", -1)) != 100:
        raise ValueError("released HARR checkpoint must record epoch 100")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("released HARR checkpoint has no model state")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("released HARR checkpoint model state is malformed")
    return state


def _install_harr_import_shims(source: Path) -> None:
    """Provide only imports needed to instantiate the released model.

    Upstream ``util.__init__`` eagerly imports optional visualization and
    geometry packages.  TextNet only needs four small symbols to construct the
    pointer branch, so shims prevent unrelated runtime dependencies from
    changing the released network or checkpoint.
    """

    source = source.resolve()
    util_path = source / "util"
    existing_util = sys.modules.get("util")
    if existing_util is not None:
        paths = {str(Path(item).resolve()) for item in getattr(existing_util, "__path__", [])}
        if str(util_path.resolve()) not in paths:
            raise RuntimeError("a conflicting top-level 'util' package is already imported")
    else:
        util_package = types.ModuleType("util")
        util_package.__path__ = [str(util_path)]
        sys.modules["util"] = util_package

    if "util.converter" not in sys.modules:
        converter = types.ModuleType("util.converter")
        codec_lines = (util_path / "codec.txt").read_text(
            encoding="utf-8",
            errors="strict",
        ).splitlines()
        if not codec_lines:
            raise ValueError("HARR codec.txt is empty")
        converter.keys = codec_lines[0]
        sys.modules["util.converter"] = converter

    if "util.misc" not in sys.modules:
        misc = types.ModuleType("util.misc")
        misc.mkdirs = lambda *_args, **_kwargs: None
        misc.to_device = lambda tensor: tensor
        sys.modules["util.misc"] = misc


def _load_harr_textnet(source: Path):
    verify_harr_source(source)
    source = source.resolve()
    _install_harr_import_shims(source)
    source_text = str(source)
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        vgg_module = importlib.import_module("network.vgg")
        module_path = Path(vgg_module.__file__).resolve()
        if source not in module_path.parents:
            raise RuntimeError(
                f"a conflicting top-level 'network' package was loaded from {module_path}"
            )

        original_vgg = vgg_module.VggNet

        class ReleasedCheckpointVgg(original_vgg):
            def __init__(self, name: str = "vgg16", pretrain: bool = True) -> None:
                # TextNet hard-codes ImageNet bootstrap even though every
                # backbone tensor is immediately replaced by the complete
                # released HARR checkpoint.
                super().__init__(name=name, pretrain=False)

        vgg_module.VggNet = ReleasedCheckpointVgg
        try:
            textnet_module = importlib.import_module("network.textnet")
        finally:
            vgg_module.VggNet = original_vgg
    finally:
        if inserted:
            sys.path.remove(source_text)
    module_path = Path(textnet_module.__file__).resolve()
    if source not in module_path.parents:
        raise RuntimeError(f"HARR TextNet was loaded from unexpected path {module_path}")
    return textnet_module.TextNet


def build_harr_pointer_model(
    source: Path,
    checkpoint_path: Path,
    *,
    checkpoint: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the full checkpoint strictly, then retain its active pointer branch."""

    source = source.resolve()
    checkpoint_path = checkpoint_path.resolve()
    commit = verify_harr_source(source)
    if checkpoint is None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("HARR checkpoint must be a mapping")
    state = verify_harr_checkpoint(checkpoint_path, checkpoint)
    textnet_class = _load_harr_textnet(source)
    full_model = textnet_class(backbone="vgg", is_training=False)
    incompatible = full_model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"HARR checkpoint/model mismatch: {incompatible}")
    full_parameters = sum(parameter.numel() for parameter in full_model.parameters())
    model = HARRPointerBranch(full_model.fpn, full_model.predict)
    active_parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "protocol": HARR_POINTER_PROTOCOL,
        "repository": HARR_REPOSITORY,
        "source_commit": commit,
        "source_license": "MIT",
        "released_checkpoint_url": HARR_RELEASED_CHECKPOINT_URL,
        "released_checkpoint_sha256": HARR_RELEASED_CHECKPOINT_SHA256,
        "released_checkpoint_epoch": 100,
        "backbone": "VGG16",
        "full_checkpoint_parameters": full_parameters,
        "active_pointer_branch_parameters": active_parameters,
        "image_size": HARR_IMAGE_SIZE,
        "crop_expansion": HARR_CROP_EXPANSION,
        "pointer_threshold": HARR_POINTER_THRESHOLD,
        "channel_order": "BGR (as in the released OpenCV inference script)",
        "training_data": "HARR authors' released recognition dataset",
        "syncg_training_images_used": 0,
        "pointer10k_training_images_used": 0,
        "comparison_scope": (
            "released HARR pointer branch with common ground-truth dial crop; "
            "not HARR detector/OCR/end-to-end scalar reading"
        ),
    }
    return model, metadata


def harr_tensor_from_bbox(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    image_size: int = HARR_IMAGE_SIZE,
    expansion: float = HARR_CROP_EXPANSION,
) -> torch.Tensor:
    """Apply the common square crop with HARR's native BGR normalization."""

    matrix = affine_for_dial(
        bbox,
        output_size=image_size,
        expansion=expansion,
    )
    crop = cv2.warpAffine(
        image,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    normalized = crop.astype(np.float32) / 255.0
    normalized -= np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    normalized /= np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(
        np.ascontiguousarray(normalized.transpose(2, 0, 1))
    )


def _first_hough_segment(
    pointer_mask: np.ndarray,
) -> tuple[float, float, float, float] | None:
    skeleton = skeletonize(pointer_mask.astype(bool))
    edges = skeleton.astype(np.uint8) * 255
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        10,
        np.array([]),
        minLineLength=10,
        maxLineGap=400,
    )
    if lines is None or len(lines) == 0:
        return None
    x1, y1, x2, y2 = lines[0, 0]
    return float(x1), float(y1), float(x2), float(y2)


def decode_harr_pointer_direction(
    pointer_logits: torch.Tensor,
    *,
    threshold: float = HARR_POINTER_THRESHOLD,
) -> HARRDirectionPrediction:
    """Reproduce HARR's released mask-to-line direction post-processing."""

    if pointer_logits.ndim != 4 or pointer_logits.shape[1] != 1:
        raise ValueError("HARR pointer logits must have shape [B, 1, H, W]")
    if not 0.0 < threshold < 1.0:
        raise ValueError("HARR pointer threshold must be between zero and one")
    device = pointer_logits.device
    probabilities = torch.sigmoid(pointer_logits).detach().float().cpu().numpy()[:, 0]
    directions = np.zeros((len(probabilities), 2), dtype=np.float32)
    confidences = np.zeros(len(probabilities), dtype=np.float32)
    valid = np.zeros(len(probabilities), dtype=bool)
    failures: list[str | None] = []
    for index, probability in enumerate(probabilities):
        mask = probability > threshold
        if not np.any(mask):
            failures.append("empty_pointer_mask")
            continue
        segment = _first_hough_segment(mask)
        if segment is None:
            failures.append("no_hough_pointer_line")
            continue
        x1, y1, x2, y2 = segment
        height, width = probability.shape
        center = np.asarray([0.5 * width, 0.5 * height], dtype=np.float32)
        first = np.asarray([x1, y1], dtype=np.float32)
        second = np.asarray([x2, y2], dtype=np.float32)
        if float(np.sum((first - center) ** 2)) <= float(
            np.sum((second - center) ** 2)
        ):
            origin, tip = first, second
        else:
            origin, tip = second, first
        direction = tip - origin
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-8:
            failures.append("collapsed_hough_pointer_line")
            continue
        directions[index] = direction / norm
        confidences[index] = float(np.mean(probability[mask]))
        valid[index] = True
        failures.append(None)
    return HARRDirectionPrediction(
        direction=torch.from_numpy(directions).to(device),
        confidence=torch.from_numpy(confidences).to(device),
        valid=torch.from_numpy(valid).to(device),
        failure_reason=tuple(failures),
    )
