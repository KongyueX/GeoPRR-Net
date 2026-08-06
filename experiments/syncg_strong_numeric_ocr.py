"""Independent mature-backbone OCR fallback for the public SyncG corpus.

The primary Tiny CTC experiment remains untouched.  This module provides a
drop-in *recognizer-only* upgrade which keeps the same signed-real vocabulary,
frozen group split, detector checkpoint and top-K CTC posterior contract used
by GARC.  Its visual encoder is initialized from the official TorchVision
MobileNetV3-Small ImageNet checkpoint and is followed by SVTR-style global
self-attention blocks.  It is intentionally a PyTorch implementation rather
than a claim that the architecture is the official PaddleOCR SVTR model.

No download occurs at import or inference time.  The caller must provide the
pretrained file and its full SHA-256 is checked before it can initialize a
training run.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Final, Mapping

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.syncg_numeric_ocr import (
    CHECKPOINT_PROTOCOL,
    IMAGE_NET_MEAN,
    IMAGE_NET_STD,
    OCRPosteriorToken,
    VOCABULARY,
    CTCStringHypothesis,
    GaugeTextDetector,
    _load_checkpoint,
    ctc_prefix_beam_search,
    detect_boxes_from_probability,
    extract_word_crop,
    greedy_ctc_decode,
    normalize_numeric_text,
    resize_recognizer_crop,
    sha256_file,
)


STRONG_CHECKPOINT_PROTOCOL: Final[str] = "syncg_public_strong_numeric_ocr_checkpoint_v1"
STRONG_ARCHITECTURE: Final[str] = "MobileNetV3SmallTruncated_SVTRStyle4_CTC"
OFFICIAL_MOBILENET_V3_SMALL_URL: Final[str] = (
    "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
)
OFFICIAL_MOBILENET_V3_SMALL_SHA256: Final[str] = (
    "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
)
DEFAULT_OFFICIAL_WEIGHTS: Final[Path] = Path(
    r"C:\pointer_read\strong_numeric_ocr\weights\mobilenet_v3_small-047dcff4.pth"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_official_backbone(path: Path) -> dict[str, Any]:
    """Fail closed unless ``path`` is the exact official TorchVision artifact."""

    resolved = Path(path).resolve(strict=True)
    digest = sha256_file(resolved)
    _require(
        digest == OFFICIAL_MOBILENET_V3_SMALL_SHA256,
        "MobileNetV3-Small pretrained artifact SHA-256 mismatch",
    )
    return {
        "source": "torchvision_mobilenet_v3_small_imagenet1k_v1",
        "url": OFFICIAL_MOBILENET_V3_SMALL_URL,
        "path": str(resolved),
        "sha256": digest,
        "license": "BSD-3-Clause (TorchVision code/model distribution)",
    }


class MobileSVTRCTCRecognizer(nn.Module):
    """MobileNetV3 visual features plus compact global sequence attention.

    Input remains ``[B,1,32,160]`` so the existing public crop dataset and GARC
    backend contract do not change.  Internally the word crop is enlarged to
    48x256, producing sixteen CTC time steps instead of Tiny CTC's shorter
    sequence.  This is important for repeated digits and signed decimals.
    """

    def __init__(
        self,
        *,
        classes: int = len(VOCABULARY),
        embedding_dim: int = 256,
        attention_heads: int = 8,
        attention_layers: int = 4,
        feedforward_dim: int = 768,
        dropout: float = 0.10,
        pretrained_backbone_path: Path | None = None,
    ) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        _require(embedding_dim % attention_heads == 0, "attention head dimension mismatch")
        base = mobilenet_v3_small(weights=None)
        self.pretrained_identity: Mapping[str, Any] = {
            "source": "random_init",
            "sha256": None,
        }
        if pretrained_backbone_path is not None:
            identity = verify_official_backbone(pretrained_backbone_path)
            state = torch.load(
                Path(pretrained_backbone_path).resolve(strict=True),
                map_location="cpu",
                weights_only=True,
            )
            _require(isinstance(state, Mapping), "official backbone state is not a mapping")
            base.load_state_dict(state, strict=True)
            self.pretrained_identity = identity

        # Blocks 0..8 retain stride 16 (48x256 -> 3x16) and are all covered by
        # the official ImageNet checkpoint.  The remaining classification-tail
        # blocks would reduce the CTC sequence to only eight positions.
        self.backbone = base.features[:9]
        self.project = nn.Sequential(
            nn.Linear(48, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, 32, embedding_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence = nn.TransformerEncoder(
            layer,
            num_layers=attention_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)

    def set_backbone_frozen(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _require(image.ndim == 4, "recognizer input must be BCHW")
        _require(image.shape[1] in (1, 3), "recognizer input must have one or three channels")
        # ``resize_recognizer_crop`` intentionally emits the Tiny-CTC contract
        # in [-1, 1].  Convert it back to image intensity before applying the
        # official ImageNet normalization expected by MobileNetV3.
        value = image.float().mul(0.5).add(0.5).clamp(0.0, 1.0)
        value = F.interpolate(
            value, size=(48, 256), mode="bilinear", align_corners=False
        )
        if value.shape[1] == 1:
            value = value.repeat(1, 3, 1, 1)
        mean = value.new_tensor(IMAGE_NET_MEAN).view(1, 3, 1, 1)
        std = value.new_tensor(IMAGE_NET_STD).view(1, 3, 1, 1)
        value = (value - mean) / std
        feature = self.backbone(value)
        _require(feature.shape[1] == 48, "MobileNetV3 feature channel drift")
        feature = feature.mean(dim=2).permute(0, 2, 1).contiguous()
        sequence = self.project(feature)
        _require(sequence.shape[1] <= self.position.shape[1], "sequence exceeds position table")
        sequence = sequence + self.position[:, : sequence.shape[1]]
        sequence = self.output_norm(self.sequence(sequence))
        return self.classifier(sequence).permute(1, 0, 2).contiguous()


def model_inventory(model: nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def load_strong_recognizer_checkpoint(path: Path) -> tuple[MobileSVTRCTCRecognizer, Mapping[str, Any]]:
    checkpoint = torch.load(
        Path(path).resolve(strict=True), map_location="cpu", weights_only=False
    )
    _require(isinstance(checkpoint, Mapping), "strong recognizer checkpoint is not a mapping")
    _require(
        checkpoint.get("protocol") == STRONG_CHECKPOINT_PROTOCOL,
        "strong recognizer checkpoint protocol drift",
    )
    _require(checkpoint.get("status") == "complete", "strong recognizer checkpoint is not formal")
    _require(checkpoint.get("component") == "recognizer", "wrong strong checkpoint component")
    _require(checkpoint.get("vocabulary") == list(VOCABULARY), "strong OCR vocabulary drift")
    config = checkpoint.get("model_config")
    _require(isinstance(config, Mapping), "strong recognizer model_config missing")
    _require(config.get("architecture") == STRONG_ARCHITECTURE, "strong architecture drift")
    initialization = checkpoint.get("initialization")
    _require(isinstance(initialization, Mapping), "strong initialization provenance missing")
    _require(
        initialization.get("sha256") == OFFICIAL_MOBILENET_V3_SMALL_SHA256,
        "strong recognizer pretrained provenance drift",
    )
    code = checkpoint.get("code")
    _require(isinstance(code, Mapping), "strong recognizer code provenance missing")
    _require(
        code.get("implementation_sha256") == sha256_file(Path(__file__)),
        "strong recognizer implementation drift",
    )
    model = MobileSVTRCTCRecognizer(
        embedding_dim=int(config["embedding_dim"]),
        attention_heads=int(config["attention_heads"]),
        attention_layers=int(config["attention_layers"]),
        feedforward_dim=int(config["feedforward_dim"]),
        dropout=float(config["dropout"]),
    )
    state = checkpoint.get("state_dict")
    _require(isinstance(state, Mapping), "strong recognizer state_dict missing")
    model.load_state_dict(state, strict=True)
    return model, checkpoint


class StrongGaugeNumericOCRBackend:
    """Current frozen detector plus the independent strong recognizer.

    It deliberately exposes the same ``infer_with_posteriors`` interface as
    :class:`GaugeNumericOCRBackend`, so GARC and its top-K consensus decoder do
    not require any code or protocol changes.
    """

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
        recognizer, recognizer_value = load_strong_recognizer_checkpoint(
            recognizer_checkpoint
        )
        self.device = torch.device(device)
        self.detector_size = int(detector_value["model_config"]["image_size"])
        geometry_channels = int(
            detector_value["model_config"].get("annular_geometry_channels", 0)
        )
        _require(geometry_channels == 0, "strong backend cannot accept caller geometry channels")
        self.detector = GaugeTextDetector(pretrained=False, annular_geometry_channels=0)
        self.detector.load_state_dict(detector_value["state_dict"], strict=True)
        self.recognizer = recognizer
        self.detector.to(self.device).eval()
        self.recognizer.to(self.device).eval()
        self.threshold = float(threshold)
        self.posterior_top_k = int(posterior_top_k)
        _require(1 <= self.posterior_top_k <= 32, "posterior_top_k is outside [1,32]")
        self.identity = {
            "backend": "syncg_gauge_detector_mobile_svtr_ctc",
            "detector_protocol": CHECKPOINT_PROTOCOL,
            "recognizer_protocol": STRONG_CHECKPOINT_PROTOCOL,
            "prediction_space": "signed_real_numeric_strings",
            "caller_supplied_boxes_allowed": False,
            "posterior_interface": "infer_with_posteriors",
            "detector": {
                "path": str(Path(detector_checkpoint).resolve(strict=True)),
                "sha256": sha256_file(detector_checkpoint),
            },
            "recognizer": {
                "path": str(Path(recognizer_checkpoint).resolve(strict=True)),
                "sha256": sha256_file(recognizer_checkpoint),
                "initialization_sha256": recognizer_value["initialization"]["sha256"],
            },
            "threshold": self.threshold,
            "posterior_top_k": self.posterior_top_k,
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
        resized = cv2.resize(
            image,
            (self.detector_size, self.detector_size),
            interpolation=cv2.INTER_AREA,
        )
        # GaugeTextDetector expects ImageNet-normalized input.  Importing the
        # helper locally keeps this module independent from detector internals.
        from experiments.syncg_numeric_ocr import detector_tensor

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
        for text, recognition_score, beam, box_value in zip(
            texts, recognition_scores, posterior_beams, original_boxes, strict=True
        ):
            x1, y1, x2, y2, detector_score = box_value
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
            tokens.append(OCRToken(text=normalized, score=score, box=box).validate())
        return tokens, posterior_tokens, time.perf_counter() - started
