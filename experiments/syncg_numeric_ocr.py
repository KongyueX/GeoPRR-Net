"""Public SyncG numeric OCR data/model primitives.

The deployment boundary is a whole canonical meter ROI.  A lightweight text
detector proposes numeric word boxes and a CTC recognizer emits variable-length
signed decimal strings.  Neither component accepts a scale value, endpoint
label, or caller-supplied text box at inference time.

Ground-truth word boxes are used only by the public ``SyncG/train`` training
datasets in this module.  The primary backend implements the ``OCRBackend``
contract from :mod:`experiments.automatic_numeric_range`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


PROTOCOL: Final[str] = "syncg_public_numeric_ocr_v1"
CHECKPOINT_PROTOCOL: Final[str] = "syncg_public_numeric_ocr_checkpoint_v1"
DEFAULT_CHARACTERS: Final[str] = "0123456789-.+"
VOCABULARY: Final[tuple[str, ...]] = ("<blank>", *tuple(DEFAULT_CHARACTERS))
BLANK_INDEX: Final[int] = 0
CHAR_TO_INDEX: Final[dict[str, int]] = {
    character: index for index, character in enumerate(VOCABULARY) if index
}
IMAGE_NET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGE_NET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_MINUS_TRANSLATION = str.maketrans(
    {"−": "-", "‒": "-", "–": "-", "—": "-", "﹣": "-", "－": "-", "＋": "+"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_numeric_text(value: Any) -> str:
    """Normalize conservative OCR punctuation while preserving real values."""

    text = str(value).strip().translate(_MINUS_TRANSLATION)
    if "," in text and "." not in text and text.count(",") == 1:
        text = text.replace(",", ".")
    if not _NUMERIC_RE.fullmatch(text):
        raise ValueError(f"not a signed real numeric string: {value!r}")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"numeric string is non-finite: {value!r}")
    unknown = sorted(set(text) - set(DEFAULT_CHARACTERS))
    if unknown:
        raise ValueError(f"numeric string contains unsupported characters: {unknown}")
    return text


def encode_text(text: str) -> list[int]:
    normalized = normalize_numeric_text(text)
    return [CHAR_TO_INDEX[character] for character in normalized]


def greedy_ctc_decode(logits: torch.Tensor) -> tuple[list[str], list[float]]:
    """Collapse CTC paths and return per-sequence geometric-mean confidence."""

    _require(logits.ndim == 3, "CTC logits must have shape [T,B,C]")
    _require(logits.shape[2] == len(VOCABULARY), "CTC vocabulary size drift")
    probability = torch.softmax(logits.float(), dim=2)
    score, index = probability.max(dim=2)
    texts: list[str] = []
    confidences: list[float] = []
    for batch_index in range(index.shape[1]):
        previous = BLANK_INDEX
        characters: list[str] = []
        retained: list[float] = []
        for step in range(index.shape[0]):
            current = int(index[step, batch_index])
            if current != BLANK_INDEX and current != previous:
                characters.append(VOCABULARY[current])
                retained.append(float(score[step, batch_index]))
            previous = current
        texts.append("".join(characters))
        if retained:
            confidences.append(float(math.exp(np.log(np.clip(retained, 1e-8, 1.0)).mean())))
        else:
            confidences.append(0.0)
    return texts, confidences


@dataclass(frozen=True)
class CTCStringHypothesis:
    text: str
    log_probability: float
    beam_probability: float


def _logaddexp(*values: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def ctc_prefix_beam_search(
    logits: torch.Tensor, *, beam_width: int = 5
) -> list[list[CTCStringHypothesis]]:
    """Return top-K CTC string posteriors for later arithmetic consensus.

    ``beam_probability`` is normalized over the retained beams.  The original
    CTC log probability is preserved so a later geometry-conditioned decoder
    can recalibrate or combine hypotheses without being restricted to top-1.
    """

    _require(logits.ndim == 3, "CTC logits must have shape [T,B,C]")
    _require(logits.shape[2] == len(VOCABULARY), "CTC vocabulary size drift")
    _require(1 <= int(beam_width) <= 64, "beam_width is outside [1,64]")
    log_probability = torch.log_softmax(logits.float(), dim=2).detach().cpu().numpy()
    outputs: list[list[CTCStringHypothesis]] = []
    for batch_index in range(log_probability.shape[1]):
        beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}
        for step in range(log_probability.shape[0]):
            next_beams: dict[str, tuple[float, float]] = {}

            def update(prefix: str, blank: float | None = None, nonblank: float | None = None) -> None:
                old_blank, old_nonblank = next_beams.get(prefix, (-math.inf, -math.inf))
                next_beams[prefix] = (
                    _logaddexp(old_blank, blank if blank is not None else -math.inf),
                    _logaddexp(old_nonblank, nonblank if nonblank is not None else -math.inf),
                )

            for prefix, (prefix_blank, prefix_nonblank) in beams.items():
                blank_score = float(log_probability[step, batch_index, BLANK_INDEX])
                update(prefix, blank=_logaddexp(prefix_blank, prefix_nonblank) + blank_score)
                for character_index in range(1, len(VOCABULARY)):
                    character = VOCABULARY[character_index]
                    score = float(log_probability[step, batch_index, character_index])
                    if prefix and character == prefix[-1]:
                        update(prefix, nonblank=prefix_nonblank + score)
                        update(prefix + character, nonblank=prefix_blank + score)
                    else:
                        update(
                            prefix + character,
                            nonblank=_logaddexp(prefix_blank, prefix_nonblank) + score,
                        )
            ranked = sorted(
                next_beams.items(),
                key=lambda item: _logaddexp(item[1][0], item[1][1]),
                reverse=True,
            )[: int(beam_width)]
            beams = dict(ranked)
        ranked_final = [
            (prefix, _logaddexp(blank, nonblank))
            for prefix, (blank, nonblank) in beams.items()
        ]
        ranked_final.sort(key=lambda item: item[1], reverse=True)
        normalization = _logaddexp(*(score for _, score in ranked_final))
        outputs.append(
            [
                CTCStringHypothesis(
                    text=prefix,
                    log_probability=float(score),
                    beam_probability=float(math.exp(score - normalization)),
                )
                for prefix, score in ranked_final
            ]
        )
    return outputs


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def grouped_three_way_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    calibration_fraction: float = 0.10,
    validation_fraction: float = 0.10,
) -> dict[str, str]:
    """Split sample identities by physical group, targeting sample fractions."""

    _require(0.0 < calibration_fraction < 0.5, "invalid calibration fraction")
    _require(0.0 < validation_fraction < 0.5, "invalid validation fraction")
    _require(calibration_fraction + validation_fraction < 1.0, "empty training fraction")
    by_group: dict[str, list[str]] = {}
    all_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        _require(bool(sample_id and group_id), "split row lacks sample_id/group_id")
        _require(sample_id not in all_ids, f"duplicate sample_id: {sample_id}")
        all_ids.add(sample_id)
        by_group.setdefault(group_id, []).append(sample_id)
    _require(len(by_group) >= 3, "at least three groups are required")

    def take_groups(candidates: set[str], target: int, salt: str) -> set[str]:
        ordered = sorted(
            candidates,
            key=lambda group: hashlib.sha256(
                f"{seed}:{salt}:{group}".encode("utf-8")
            ).digest(),
        )
        selected: set[str] = set()
        count = 0
        for group in ordered:
            if count >= target and selected:
                break
            if len(candidates) - len(selected) <= 1:
                break
            selected.add(group)
            count += len(by_group[group])
        _require(bool(selected), f"{salt} partition is empty")
        return selected

    total = len(rows)
    groups = set(by_group)
    validation_groups = take_groups(
        groups, max(1, round(total * validation_fraction)), "validation"
    )
    remaining = groups - validation_groups
    calibration_groups = take_groups(
        remaining, max(1, round(total * calibration_fraction)), "calibration"
    )
    train_groups = remaining - calibration_groups
    _require(bool(train_groups), "training partition is empty")
    _require(
        not train_groups.intersection(calibration_groups | validation_groups)
        and not calibration_groups.intersection(validation_groups),
        "group leakage in OCR split",
    )
    assignment: dict[str, str] = {}
    for group, sample_ids in by_group.items():
        partition = (
            "validation"
            if group in validation_groups
            else "calibration"
            if group in calibration_groups
            else "train"
        )
        assignment.update({sample_id: partition for sample_id in sample_ids})
    _require(set(assignment) == all_ids, "split assignment inventory drift")
    return assignment


def canonical_roi_bounds(
    image_shape: Sequence[int], bbox_xyxy: Sequence[float]
) -> tuple[int, int, int, int]:
    _require(len(image_shape) >= 2 and len(bbox_xyxy) >= 4, "invalid image/bbox shape")
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy[:4])
    _require(
        width >= 2
        and height >= 2
        and np.isfinite([x1, y1, x2, y2]).all()
        and x2 > x1
        and y2 > y1,
        "invalid dial bbox",
    )
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    return left, top, right, bottom


def normalize_bbox_to_roi(
    bbox_xyxy: Sequence[float], roi_bounds: Sequence[int]
) -> tuple[float, float, float, float]:
    _require(len(bbox_xyxy) >= 4 and len(roi_bounds) == 4, "invalid bbox/bounds")
    left, top, right, bottom = (float(value) for value in roi_bounds)
    width, height = right - left, bottom - top
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy[:4])
    normalized = np.asarray(
        [(x1 - left) / width, (y1 - top) / height, (x2 - left) / width, (y2 - top) / height],
        dtype=np.float64,
    )
    _require(np.isfinite(normalized).all(), "non-finite normalized text bbox")
    normalized = np.clip(normalized, 0.0, 1.0)
    _require(
        normalized[2] - normalized[0] > 1e-6
        and normalized[3] - normalized[1] > 1e-6,
        "text bbox lies outside canonical dial ROI",
    )
    return tuple(float(value) for value in normalized)


def canonical_tight_roi(
    image_bgr: np.ndarray, bbox_xyxy: Sequence[float], *, output_size: int
) -> np.ndarray:
    _require(
        isinstance(image_bgr, np.ndarray)
        and image_bgr.dtype == np.uint8
        and image_bgr.ndim == 3
        and image_bgr.shape[2] == 3,
        "expected uint8 BGR image",
    )
    left, top, right, bottom = canonical_roi_bounds(image_bgr.shape, bbox_xyxy)
    crop = image_bgr[top:bottom, left:right]
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > output_size else cv2.INTER_LINEAR
    return cv2.resize(crop, (output_size, output_size), interpolation=interpolation)


def render_detection_target(
    boxes_normalized: Sequence[Sequence[float]],
    *,
    size: int,
    shrink_fraction: float = 0.08,
) -> np.ndarray:
    _require(size >= 32, "detector target is too small")
    target = np.zeros((size, size), dtype=np.float32)
    for box in boxes_normalized:
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        dx = (x2 - x1) * shrink_fraction
        dy = (y2 - y1) * shrink_fraction
        px1 = int(np.clip(round((x1 + dx) * (size - 1)), 0, size - 1))
        py1 = int(np.clip(round((y1 + dy) * (size - 1)), 0, size - 1))
        px2 = int(np.clip(round((x2 - dx) * (size - 1)), px1, size - 1))
        py2 = int(np.clip(round((y2 - dy) * (size - 1)), py1, size - 1))
        cv2.rectangle(target, (px1, py1), (px2, py2), 1.0, thickness=-1)
    return target


def extract_word_crop(
    image_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
    *,
    context_fraction: float = 0.20,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy[:4])
    box_height = max(1.0, y2 - y1)
    box_width = max(1.0, x2 - x1)
    pad_x = max(1.0, box_height * context_fraction, box_width * 0.04)
    pad_y = max(1.0, box_height * context_fraction)
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    _require(right > left and bottom > top, "word crop collapsed")
    return cv2.cvtColor(image_bgr[top:bottom, left:right], cv2.COLOR_BGR2GRAY)


def synthesize_decimal_word(
    image_gray: np.ndarray, label: str, rng: np.random.Generator
) -> tuple[np.ndarray, str, bool]:
    """Insert a decimal dot into an integer crop without external imagery."""

    normalized = normalize_numeric_text(label)
    if "." in normalized:
        return image_gray, normalized, False
    sign = normalized[0] if normalized and normalized[0] in "+-" else ""
    digits = normalized[len(sign) :]
    if len(digits) < 2 or not digits.isdigit():
        return image_gray, normalized, False
    position = int(rng.integers(1, len(digits)))
    height, width = image_gray.shape[:2]
    character_units = len(digits) + (0.55 if sign else 0.0)
    prefix_units = position + (0.55 if sign else 0.0)
    split_x = int(np.clip(round(width * prefix_units / character_units), 1, width - 1))
    gap = max(2, int(round(height * 0.12)))
    border = np.concatenate(
        (image_gray[0], image_gray[-1], image_gray[:, 0], image_gray[:, -1])
    )
    background = int(round(float(np.median(border))))
    low, high = np.percentile(image_gray, (5, 95))
    foreground = int(round(low if abs(background - low) >= abs(background - high) else high))
    canvas = np.full((height, width + gap), background, dtype=np.uint8)
    canvas[:, :split_x] = image_gray[:, :split_x]
    canvas[:, split_x + gap :] = image_gray[:, split_x:]
    radius = max(1, int(round(height * 0.045)))
    cv2.circle(
        canvas,
        (split_x + gap // 2, int(round(height * 0.78))),
        radius,
        foreground,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    decimal = f"{sign}{digits[:position]}.{digits[position:]}"
    return canvas, decimal, True


def resize_recognizer_crop(
    image_gray: np.ndarray, *, height: int = 32, width: int = 160
) -> torch.Tensor:
    _require(image_gray.ndim == 2 and image_gray.size > 0, "invalid grayscale word crop")
    scale = min(height / image_gray.shape[0], width / image_gray.shape[1])
    resized_width = max(1, min(width, int(round(image_gray.shape[1] * scale))))
    resized_height = max(1, min(height, int(round(image_gray.shape[0] * scale))))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(image_gray, (resized_width, resized_height), interpolation=interpolation)
    border = np.concatenate((resized[0], resized[-1], resized[:, 0], resized[:, -1]))
    fill = int(round(float(np.median(border))))
    canvas = np.full((height, width), fill, dtype=np.uint8)
    y0 = (height - resized_height) // 2
    canvas[y0 : y0 + resized_height, :resized_width] = resized
    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)[None]
    return (tensor - 0.5) / 0.5


def detector_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(IMAGE_NET_MEAN, dtype=tensor.dtype)[:, None, None]
    std = torch.tensor(IMAGE_NET_STD, dtype=tensor.dtype)[:, None, None]
    return (tensor - mean) / std


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"{path}:{line_number}: row is not an object")
            rows.append(value)
    _require(bool(rows), f"empty JSONL: {path}")
    return rows


@dataclass(frozen=True)
class OCRCorpus:
    root: Path
    samples: tuple[Mapping[str, Any], ...]
    tokens: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]

    @classmethod
    def load(cls, root: Path) -> "OCRCorpus":
        resolved = Path(root).resolve(strict=True)
        summary_path = resolved / "summary.json"
        samples_path = resolved / "samples.jsonl"
        tokens_path = resolved / "tokens.jsonl"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _require(summary.get("protocol") == PROTOCOL, "OCR corpus protocol drift")
        _require(summary.get("status") == "complete", "OCR corpus is incomplete")
        _require(sha256_file(samples_path) == summary["artifacts"]["samples_sha256"], "sample hash drift")
        _require(sha256_file(tokens_path) == summary["artifacts"]["tokens_sha256"], "token hash drift")
        samples = _load_jsonl(samples_path)
        tokens = _load_jsonl(tokens_path)
        _require(len(samples) == int(summary["inventory"]["samples"]), "sample count drift")
        _require(len(tokens) == int(summary["inventory"]["tokens"]), "token count drift")
        return cls(resolved, tuple(samples), tuple(tokens), summary)

    def partition_samples(self, partition: str) -> list[Mapping[str, Any]]:
        rows = [row for row in self.samples if row.get("partition") == partition]
        _require(bool(rows), f"empty sample partition: {partition}")
        return rows

    def partition_tokens(self, partition: str) -> list[Mapping[str, Any]]:
        rows = [row for row in self.tokens if row.get("partition") == partition]
        _require(bool(rows), f"empty token partition: {partition}")
        return rows


def _resolve_public_relative(project_root: Path, value: Any, *, kind: str) -> Path:
    root = Path(project_root).resolve(strict=True)
    allowed = (root / f"datasets/SyncG/syncG/{kind}/train").resolve(strict=True)
    candidate = (root / str(value)).resolve(strict=True)
    try:
        candidate.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"OCR {kind} path escapes public SyncG/train: {candidate}") from error
    return candidate


class SyncGTextDetectionDataset(Dataset):
    def __init__(
        self,
        corpus: OCRCorpus,
        *,
        project_root: Path,
        partition: str,
        image_size: int = 512,
        training: bool = False,
        seed: int = 0,
        limit: int | None = None,
    ) -> None:
        rows = corpus.partition_samples(partition)
        self.rows = tuple(rows[:limit] if limit else rows)
        self.project_root = Path(project_root)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.seed = int(seed)
        self.epoch = 0
        _require(self.image_size >= 64, "detector image_size is too small")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = _resolve_public_relative(
            self.project_root, row["image_path"], kind="images"
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        _require(image is not None, f"failed to read public image: {row['sample_id']}")
        roi = canonical_tight_roi(image, row["dial_bbox"], output_size=self.image_size)
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 103)
            value = roi.astype(np.float32)
            if rng.random() < 0.75:
                value = value * float(rng.uniform(0.65, 1.40)) + float(rng.uniform(-35, 35))
            value = np.clip(value, 0, 255)
            if rng.random() < 0.35:
                value = cv2.GaussianBlur(value, (0, 0), sigmaX=float(rng.uniform(0.2, 1.4)))
            if rng.random() < 0.25:
                value += rng.normal(0.0, float(rng.uniform(1.0, 9.0)), value.shape)
            roi = np.clip(value, 0, 255).astype(np.uint8)
            if rng.random() < 0.30:
                quality = int(rng.integers(50, 95))
                ok, encoded = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, quality])
                _require(bool(ok), "detector JPEG augmentation failed")
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                _require(decoded is not None, "detector JPEG decode failed")
                roi = decoded
        boxes = [token["bbox_roi_normalized"] for token in row["tokens"]]
        target = render_detection_target(boxes, size=self.image_size)
        return {
            "image": detector_tensor(roi),
            "target": torch.from_numpy(target)[None],
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "sample_id": str(row["sample_id"]),
            "group_id": str(row["group_id"]),
        }


class SyncGTextRecognitionDataset(Dataset):
    def __init__(
        self,
        corpus: OCRCorpus,
        *,
        project_root: Path,
        partition: str,
        seed: int,
        training: bool,
        decimal_probability: float = 0.20,
        limit: int | None = None,
    ) -> None:
        rows = corpus.partition_tokens(partition)
        self.rows = tuple(rows[:limit] if limit else rows)
        self.project_root = Path(project_root)
        self.seed = int(seed)
        self.training = bool(training)
        self.decimal_probability = float(decimal_probability if training else 0.0)
        _require(0.0 <= self.decimal_probability <= 1.0, "invalid decimal probability")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = _resolve_public_relative(
            self.project_root, row["image_path"], kind="images"
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        _require(image is not None, f"failed to read public image: {row['sample_id']}")
        crop = extract_word_crop(image, row["bbox_original"])
        label = normalize_numeric_text(row["text"])
        synthesized = False
        if self.training:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 101)
            if rng.random() < self.decimal_probability:
                crop, label, synthesized = synthesize_decimal_word(crop, label, rng)
            if rng.random() < 0.35:
                sigma = float(rng.uniform(0.15, 1.15))
                crop = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma)
            if rng.random() < 0.45:
                gain = float(rng.uniform(0.70, 1.35))
                offset = float(rng.uniform(-25.0, 25.0))
                crop = np.clip(crop.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
        encoded = torch.tensor(encode_text(label), dtype=torch.long)
        return {
            "image": resize_recognizer_crop(crop),
            "target": encoded,
            "target_length": torch.tensor(len(encoded), dtype=torch.long),
            "text": label,
            "sample_id": str(row["sample_id"]),
            "group_id": str(row["group_id"]),
            "decimal_synthesized": torch.tensor(synthesized),
        }


class SyncGTextRecognitionImageDataset(Dataset):
    """Decode each public source image once and emit all of its word crops."""

    def __init__(
        self,
        corpus: OCRCorpus,
        *,
        project_root: Path,
        partition: str,
        seed: int,
        training: bool,
        decimal_probability: float = 0.20,
        limit: int | None = None,
    ) -> None:
        samples = corpus.partition_samples(partition)
        self.samples = tuple(samples[:limit] if limit else samples)
        tokens_by_sample: dict[str, list[Mapping[str, Any]]] = {}
        for token in corpus.partition_tokens(partition):
            tokens_by_sample.setdefault(str(token["sample_id"]), []).append(token)
        self.tokens_by_sample = {
            sample_id: tuple(sorted(values, key=lambda row: str(row["token_id"])))
            for sample_id, values in tokens_by_sample.items()
        }
        for sample in self.samples:
            _require(
                bool(self.tokens_by_sample.get(str(sample["sample_id"]))),
                f"sample has no recognition tokens: {sample['sample_id']}",
            )
        self.project_root = Path(project_root)
        self.seed = int(seed)
        self.training = bool(training)
        self.decimal_probability = float(decimal_probability if training else 0.0)
        _require(0.0 <= self.decimal_probability <= 1.0, "invalid decimal probability")
        self.epoch = 0
        self.token_count = sum(
            len(self.tokens_by_sample[str(sample["sample_id"])]) for sample in self.samples
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        sample = self.samples[index]
        image_path = _resolve_public_relative(
            self.project_root, sample["image_path"], kind="images"
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        _require(image is not None, f"failed to read public image: {sample['sample_id']}")
        rows: list[dict[str, Any]] = []
        for token_index, token in enumerate(self.tokens_by_sample[str(sample["sample_id"])]):
            crop = extract_word_crop(image, token["bbox_original"])
            label = normalize_numeric_text(token["text"])
            synthesized = False
            if self.training:
                rng = np.random.default_rng(
                    self.seed
                    + self.epoch * 1_000_003
                    + index * 1009
                    + token_index * 101
                )
                if rng.random() < self.decimal_probability:
                    crop, label, synthesized = synthesize_decimal_word(crop, label, rng)
                if rng.random() < 0.35:
                    crop = cv2.GaussianBlur(
                        crop, (0, 0), sigmaX=float(rng.uniform(0.15, 1.15))
                    )
                if rng.random() < 0.45:
                    crop = np.clip(
                        crop.astype(np.float32) * float(rng.uniform(0.70, 1.35))
                        + float(rng.uniform(-25.0, 25.0)),
                        0,
                        255,
                    ).astype(np.uint8)
            encoded = torch.tensor(encode_text(label), dtype=torch.long)
            rows.append(
                {
                    "image": resize_recognizer_crop(crop),
                    "target": encoded,
                    "target_length": torch.tensor(len(encoded), dtype=torch.long),
                    "text": label,
                    "sample_id": str(sample["sample_id"]),
                    "group_id": str(sample["group_id"]),
                    "decimal_synthesized": torch.tensor(synthesized),
                }
            )
        return rows


def recognition_collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in rows]),
        "targets": torch.cat([row["target"] for row in rows]),
        "target_lengths": torch.stack([row["target_length"] for row in rows]),
        "texts": [str(row["text"]) for row in rows],
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "decimal_synthesized": torch.stack([row["decimal_synthesized"] for row in rows]),
    }


def recognition_image_collate(
    image_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows = [row for values in image_rows for row in values]
    _require(bool(rows), "empty recognition image batch")
    return recognition_collate(rows)


class _ConvNormAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if output_channels >= 8 else 1, output_channels),
            nn.SiLU(inplace=True),
        )


class GaugeTextDetector(nn.Module):
    """MobileNetV3-small FPN for tiny whole-word text occupancy."""

    def __init__(
        self,
        *,
        pretrained: bool = False,
        fpn_channels: int = 48,
        annular_geometry_channels: int = 0,
    ) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features
        self.pretrained_identity = (
            "torchvision_mobilenet_v3_small_imagenet1k_v1" if pretrained else "random_init"
        )
        self.annular_geometry_channels = int(annular_geometry_channels)
        _require(self.annular_geometry_channels >= 0, "negative annular geometry channel count")
        channels = (16, 16, 24, 48, 576)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(value, fpn_channels, 1) for value in channels]
        )
        self.refine = nn.ModuleList(
            [_ConvNormAct(fpn_channels, fpn_channels) for _ in range(4)]
        )
        self.geometry_projection = (
            nn.Conv2d(self.annular_geometry_channels, fpn_channels, 1, bias=False)
            if self.annular_geometry_channels
            else None
        )
        self.output = nn.Sequential(
            _ConvNormAct(fpn_channels, 32),
            nn.Conv2d(32, 1, 1),
        )

    def set_early_encoder_frozen(self, frozen: bool) -> None:
        for index, block in enumerate(self.encoder):
            for parameter in block.parameters():
                parameter.requires_grad = not frozen or index >= 4

    def forward(
        self, image: torch.Tensor, annular_geometry: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.annular_geometry_channels:
            _require(annular_geometry is not None, "configured detector requires annular geometry")
            _require(
                annular_geometry.ndim == 4
                and annular_geometry.shape[0] == image.shape[0]
                and annular_geometry.shape[1] == self.annular_geometry_channels,
                "annular geometry tensor shape drift",
            )
        else:
            _require(annular_geometry is None, "baseline detector does not consume geometry channels")
        features: list[torch.Tensor] = []
        value = image
        capture = {0: 0, 1: 1, 3: 2, 8: 3, 12: 4}
        for index, block in enumerate(self.encoder):
            value = block(value)
            if index in capture:
                features.append(self.lateral[capture[index]](value))
        _require(len(features) == 5, "MobileNet feature inventory drift")
        if self.geometry_projection is not None:
            geometry = F.interpolate(
                annular_geometry.float(), size=features[0].shape[-2:], mode="bilinear", align_corners=False
            )
            features[0] = features[0] + self.geometry_projection(geometry)
        value = features[-1]
        for level, refinement in zip(range(3, -1, -1), self.refine, strict=True):
            value = F.interpolate(value, size=features[level].shape[-2:], mode="bilinear", align_corners=False)
            value = refinement(value + features[level])
        value = F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(value)


class TinyCTCRecognizer(nn.Module):
    """Small grayscale CRNN with a real-valued numeric character vocabulary."""

    def __init__(self, *, classes: int = len(VOCABULARY), hidden: int = 96) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvNormAct(1, 32),
            nn.MaxPool2d(2, 2),
            _ConvNormAct(32, 64),
            nn.MaxPool2d(2, 2),
            _ConvNormAct(64, 128),
            nn.MaxPool2d((2, 1), (2, 1)),
            _ConvNormAct(128, 192),
            nn.MaxPool2d((4, 1), (4, 1)),
        )
        self.sequence = nn.GRU(
            input_size=192,
            hidden_size=hidden,
            num_layers=2,
            dropout=0.10,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * hidden, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.features(image)
        _require(feature.shape[2] == 1, "recognizer feature height did not collapse")
        sequence = feature.squeeze(2).permute(2, 0, 1).contiguous()
        encoded, _ = self.sequence(sequence)
        return self.classifier(encoded)


def text_detection_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require(logits.shape == target.shape, "detector logits/target shape mismatch")
    positive = target.sum().clamp_min(1.0)
    negative = target.numel() - positive
    positive_weight = (negative / positive).clamp(1.0, 40.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def detect_boxes_from_probability(
    probability: np.ndarray,
    *,
    threshold: float = 0.40,
    min_area: int = 4,
) -> list[tuple[int, int, int, int, float]]:
    _require(probability.ndim == 2, "detector probability must be 2D")
    mask = (probability >= float(threshold)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int, float]] = []
    height, width = probability.shape
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < min_area or w < 2 or h < 2:
            continue
        pad_x = max(1, int(round(w * 0.12)))
        pad_y = max(1, int(round(h * 0.18)))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        score = float(probability[labels == index].mean())
        boxes.append((x1, y1, x2, y2, score))
    return sorted(boxes, key=lambda item: (item[1], item[0]))


def _load_checkpoint(path: Path, *, component: str) -> Mapping[str, Any]:
    checkpoint = torch.load(Path(path).resolve(strict=True), map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), f"{component} checkpoint is not a mapping")
    _require(checkpoint.get("protocol") == CHECKPOINT_PROTOCOL, f"{component} protocol drift")
    _require(checkpoint.get("status") == "complete", f"{component} checkpoint is not formal")
    _require(checkpoint.get("component") == component, f"wrong {component} checkpoint")
    _require(isinstance(checkpoint.get("state_dict"), Mapping), f"{component} state_dict missing")
    _require(checkpoint.get("vocabulary") == list(VOCABULARY), "OCR vocabulary drift")
    return checkpoint


@dataclass(frozen=True)
class OCRPosteriorToken:
    box: tuple[tuple[float, float], ...]
    detector_score: float
    hypotheses: tuple[CTCStringHypothesis, ...]


class GaugeNumericOCRBackend:
    """Label-free detector/CTC backend for ``AutomaticNumericRangePipeline``."""

    def __init__(
        self,
        detector_checkpoint: Path,
        recognizer_checkpoint: Path,
        *,
        device: str = "cpu",
        threshold: float = 0.40,
        posterior_top_k: int = 5,
    ) -> None:
        detector_value = _load_checkpoint(detector_checkpoint, component="detector")
        recognizer_value = _load_checkpoint(recognizer_checkpoint, component="recognizer")
        self.device = torch.device(device)
        self.detector_size = int(detector_value["model_config"]["image_size"])
        geometry_channels = int(detector_value["model_config"].get("annular_geometry_channels", 0))
        _require(geometry_channels == 0, "baseline backend cannot accept caller geometry channels")
        self.detector = GaugeTextDetector(
            pretrained=False, annular_geometry_channels=geometry_channels
        )
        self.detector.load_state_dict(detector_value["state_dict"], strict=True)
        self.recognizer = TinyCTCRecognizer()
        self.recognizer.load_state_dict(recognizer_value["state_dict"], strict=True)
        self.detector.to(self.device).eval()
        self.recognizer.to(self.device).eval()
        self.threshold = float(threshold)
        self.posterior_top_k = int(posterior_top_k)
        _require(1 <= self.posterior_top_k <= 32, "posterior_top_k is outside [1,32]")
        self.identity = {
            "backend": "syncg_gauge_numeric_detector_ctc",
            "protocol": PROTOCOL,
            "prediction_space": "signed_real_numeric_strings",
            "caller_supplied_boxes_allowed": False,
            "detector": {
                "path": str(Path(detector_checkpoint).resolve(strict=True)),
                "sha256": sha256_file(detector_checkpoint),
            },
            "recognizer": {
                "path": str(Path(recognizer_checkpoint).resolve(strict=True)),
                "sha256": sha256_file(recognizer_checkpoint),
            },
            "threshold": self.threshold,
            "posterior_top_k": self.posterior_top_k,
            "posterior_interface": "infer_with_posteriors",
            "annular_geometry_extension_channels": geometry_channels,
            "device": str(self.device),
        }

    @torch.inference_mode()
    def infer(self, image_bgr: np.ndarray) -> tuple[list[Any], float]:
        tokens, _, elapsed = self.infer_with_posteriors(image_bgr)
        return tokens, elapsed

    @torch.inference_mode()
    def infer_with_posteriors(
        self, image_bgr: np.ndarray
    ) -> tuple[list[Any], list[OCRPosteriorToken], float]:
        from experiments.automatic_numeric_range import OCRToken, validate_canonical_roi

        started = time.perf_counter()
        image = validate_canonical_roi(image_bgr)
        resized = cv2.resize(image, (self.detector_size, self.detector_size), interpolation=cv2.INTER_AREA)
        logits = self.detector(detector_tensor(resized)[None].to(self.device))
        probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
        boxes = detect_boxes_from_probability(probability, threshold=self.threshold)
        if not boxes:
            return [], [], time.perf_counter() - started
        scale_x = image.shape[1] / self.detector_size
        scale_y = image.shape[0] / self.detector_size
        crops: list[torch.Tensor] = []
        original_boxes: list[tuple[float, float, float, float, float]] = []
        for x1, y1, x2, y2, score in boxes:
            original = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            crops.append(resize_recognizer_crop(extract_word_crop(image, original)))
            original_boxes.append((*original, score))
        recognizer_logits = self.recognizer(torch.stack(crops).to(self.device))
        texts, recognition_scores = greedy_ctc_decode(recognizer_logits.cpu())
        posterior_beams = ctc_prefix_beam_search(
            recognizer_logits.cpu(), beam_width=self.posterior_top_k
        )
        tokens: list[Any] = []
        posterior_tokens: list[OCRPosteriorToken] = []
        for text, recognition_score, beam, (x1, y1, x2, y2, detector_score) in zip(
            texts, recognition_scores, posterior_beams, original_boxes, strict=True
        ):
            valid_hypotheses: list[CTCStringHypothesis] = []
            for hypothesis in beam:
                try:
                    normalized_hypothesis = normalize_numeric_text(hypothesis.text)
                except ValueError:
                    continue
                valid_hypotheses.append(
                    CTCStringHypothesis(
                        text=normalized_hypothesis,
                        log_probability=hypothesis.log_probability,
                        beam_probability=hypothesis.beam_probability,
                    )
                )
            box = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
            posterior_tokens.append(
                OCRPosteriorToken(
                    box=box,
                    detector_score=float(detector_score),
                    hypotheses=tuple(valid_hypotheses),
                )
            )
            try:
                normalized = normalize_numeric_text(text)
            except ValueError:
                if not valid_hypotheses:
                    continue
                normalized = valid_hypotheses[0].text
            score = float(math.sqrt(max(0.0, detector_score * recognition_score)))
            tokens.append(
                OCRToken(
                    text=normalized,
                    score=score,
                    box=box,
                ).validate()
            )
        return tokens, posterior_tokens, time.perf_counter() - started
