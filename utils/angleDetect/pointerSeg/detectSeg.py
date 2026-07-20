import os
import sys

filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

try:
    from .u2netp import U2NETP
except ImportError:  # Legacy execution with angleDetect directly on sys.path.
    from pointerSeg.u2netp import U2NETP

try:
    from ..dataloader import Letterbox
except ImportError:  # Legacy execution with angleDetect directly on sys.path.
    from dataloader import Letterbox


def load_u2net_state_dict(weights, map_location):
    """Load legacy raw weights or the reproducible training checkpoint format."""
    try:
        checkpoint = torch.load(
            weights,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:  # PyTorch < 2.0 has no weights_only argument.
        checkpoint = torch.load(weights, map_location=map_location)

    metadata = {}
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
        and isinstance(checkpoint["state_dict"], dict)
    ):
        state_dict = checkpoint["state_dict"]
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"state_dict", "optimizer_state_dict", "scheduler_state_dict"}
        }
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"unsupported U2Net checkpoint payload: {type(state_dict)!r}")
    return state_dict, metadata


class u2netpSeg():
    INPUT_SIZE = 256

    def __init__(self, weights, device, probability_threshold=None):
        self.weights = weights
        self.device  = device
        self.model = U2NETP().to(device)
        state_dict, checkpoint_metadata = load_u2net_state_dict(weights, device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        embedded_threshold = checkpoint_metadata.get("probability_threshold")
        self.probability_threshold = (
            probability_threshold
            if probability_threshold is not None
            else embedded_threshold
        )
        if self.probability_threshold is not None:
            self.probability_threshold = float(self.probability_threshold)
            if not 0.0 < self.probability_threshold < 1.0:
                raise ValueError(
                    "probability_threshold must be strictly between 0 and 1"
                )
        self.checkpoint_metadata = checkpoint_metadata
        self.last_probability_summary = {}
        self.transformRgb = transforms.Compose([
            Letterbox(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _imgProcess(self,img):
        img = self.transformRgb(img).unsqueeze(0).to(self.device)
        return img

    @classmethod
    def _restore_letterbox_mask(cls, mask, original_size):
        """Remove Letterbox padding before restoring the original coordinates."""
        original_width, original_height = map(int, original_size)
        if original_width <= 0 or original_height <= 0:
            raise ValueError(f"invalid original image size: {original_size}")

        ratio = min(
            cls.INPUT_SIZE / original_width,
            cls.INPUT_SIZE / original_height,
        )
        resized_width = max(1, int(original_width * ratio))
        resized_height = max(1, int(original_height * ratio))
        left = (cls.INPUT_SIZE - resized_width) // 2
        top = (cls.INPUT_SIZE - resized_height) // 2
        cropped = mask[
            top : top + resized_height,
            left : left + resized_width,
        ]
        if cropped.size == 0:
            raise ValueError(
                f"empty letterbox crop for size {original_size}: "
                f"{left=}, {top=}, {resized_width=}, {resized_height=}"
            )
        return cv2.resize(
            cropped,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )

    @staticmethod
    def _summarize_probability(probability, threshold):
        values = np.asarray(probability, dtype=np.float32)
        foreground = values >= float(threshold)
        return {
            "probability_max": float(np.max(values)),
            "probability_p99": float(np.percentile(values, 99)),
            "probability_mean": float(np.mean(values)),
            "foreground_ratio": float(np.mean(foreground)),
            "threshold": float(threshold),
        }

    def Inference(self, img):
        original_size = img.size
        img = self._imgProcess(img)
        with torch.inference_mode():
            endImg = self.model(img)
        probability = endImg[0].cpu().squeeze().numpy()
        probability = self._restore_letterbox_mask(probability, original_size)
        probability = np.clip(probability, 0.0, 1.0)
        effective_threshold = (
            self.probability_threshold
            if self.probability_threshold is not None
            else 1.0 / 255.0
        )
        self.last_probability_summary = self._summarize_probability(
            probability,
            effective_threshold,
        )
        if self.probability_threshold is None:
            # Preserve the legacy model's effective threshold (1 / 255).
            endImg = (probability * 255).astype(np.uint8)
        else:
            endImg = (
                probability >= self.probability_threshold
            ).astype(np.uint8) * 255
        return endImg
    
if __name__ == "__main__":
    img1_path = "demo.png"
    img = Image.open(img1_path).convert('RGB')

    weights = "pointerSeg\\resultSeg\\best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u2netSegModel = u2netpSeg(weights, device)

    img = u2netSegModel.Inference(img)
    
    cv2.imshow("demo", img)
    cv2.waitKey(0)
    
