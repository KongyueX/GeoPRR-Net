# -*- coding: utf-8 -*-
"""Runtime loader for the learned geometry + residual hybrid backend."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_DIR
    / "improvement_artifacts_20260709"
    / "reports"
    / "previous_pointer_residual_hybrid_training"
    / "20260713_102310"
    / "best_model.pt"
)
PERIOD = 100


def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _resize_gray(image, image_size, binary=False):
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_AREA
    image = cv2.resize(image, (image_size, image_size), interpolation=interpolation)
    image = image.astype(np.float32) / 255.0
    if binary:
        image = (image > 0.2).astype(np.float32)
    return image


def _skeletonize_binary(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary.copy()
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
    return skel


def circular_wrap_unit(v):
    return float(v) % 1.0


class ResidualHybridNet(nn.Module):
    def __init__(self, scalar_dim=6, classes=PERIOD, delta_limit=0.12):
        super().__init__()
        self.delta_limit = float(delta_limit)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(192 + 32, 160),
            nn.SiLU(inplace=True),
            nn.Dropout(0.12),
            nn.Linear(160, 96),
            nn.SiLU(inplace=True),
        )
        self.delta_head = nn.Linear(96, 1)
        self.cls_head = nn.Linear(96, classes)
        self.conf_head = nn.Linear(96, 1)

    def forward(self, x, scalar):
        z_img = self.features(x).flatten(1)
        z_scalar = self.scalar_branch(scalar)
        z = self.fuse(torch.cat([z_img, z_scalar], dim=1))
        delta = torch.tanh(self.delta_head(z)) * self.delta_limit
        logits = self.cls_head(z)
        confidence = torch.sigmoid(self.conf_head(z))
        return delta, logits, confidence


class ResidualHybridBackend:
    def __init__(self, model_path=None, device="cpu", image_size=160):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.device = torch.device(device or "cpu")
        self.image_size = int(image_size)
        self.model = None
        self.load_error = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            self.load_error = f"hybrid model not found: {self.model_path}"
            logger.warning(self.load_error)
            return

        try:
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            delta_limit = 0.12
            if isinstance(checkpoint, dict):
                delta_limit = float(checkpoint.get("args", {}).get("delta_limit", delta_limit))
            model = ResidualHybridNet(delta_limit=delta_limit).to(self.device)
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self.model = model
            self.load_error = None
            logger.info(f"hybrid residual backend loaded: {self.model_path}")
        except Exception as exc:
            self.model = None
            self.load_error = f"failed to load hybrid backend: {exc}"
            logger.warning(self.load_error)

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _prepare_inputs(self, crop_bgr, pointer_mask, geometry_reading):
        if crop_bgr is None or pointer_mask is None:
            return None, None, "crop or mask is None"

        if len(crop_bgr.shape) == 2:
            crop_gray = crop_bgr
        else:
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        crop_gray = _resize_gray(crop_gray, self.image_size, binary=False)

        if len(pointer_mask.shape) == 3:
            mask_gray = cv2.cvtColor(pointer_mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = pointer_mask
        mask_gray = _resize_gray(mask_gray, self.image_size, binary=True)
        skeleton = _skeletonize_binary((mask_gray > 0).astype(np.uint8) * 255)
        skeleton = _resize_gray(skeleton, self.image_size, binary=True)

        x = np.stack([crop_gray, mask_gray, skeleton], axis=0).astype(np.float32)

        p_geom = geometry_reading.get("progress_ratio")
        if p_geom is None:
            p_geom = geometry_reading.get("endNum_float")
            if p_geom is not None:
                p_geom = float(p_geom) / 100.0
        p_geom = self._safe_float(p_geom, 0.0)
        p_geom = min(max(p_geom, 0.0), 1.0)

        tip_info = geometry_reading.get("tip_info") or {}
        center_distance = self._safe_float(tip_info.get("center_distance"), 0.0)
        point_count = self._safe_float(tip_info.get("point_count"), 0.0)
        geom_angle = self._safe_float(geometry_reading.get("pointer_angle"), 0.0)

        scalar = np.asarray(
            [
                math.cos(2.0 * math.pi * p_geom),
                math.sin(2.0 * math.pi * p_geom),
                center_distance / 200.0,
                point_count / 5000.0,
                math.cos(math.radians(geom_angle)),
                math.sin(math.radians(geom_angle)),
            ],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(x).unsqueeze(0).to(self.device),
            torch.from_numpy(scalar).unsqueeze(0).to(self.device),
            {
                "p_geom": p_geom,
                "center_distance": center_distance,
                "point_count": point_count,
                "geom_angle": geom_angle,
            },
        )

    @staticmethod
    def _progress_to_result(progress_ratio, scale_start, scale_end, snap_out_of_range_pointer=True):
        if scale_end is None or scale_start is None:
            return None
        scale_start = float(scale_start)
        scale_end = float(scale_end)
        progress_ratio = float(progress_ratio)
        result = scale_start + progress_ratio * (scale_end - scale_start)
        low_scale = min(scale_start, scale_end)
        high_scale = max(scale_start, scale_end)
        if snap_out_of_range_pointer and (result < low_scale or result > high_scale):
            if progress_ratio <= 0.5:
                return scale_start
            return scale_end
        if result < low_scale:
            return low_scale
        if result > high_scale:
            return high_scale
        return result

    def predict(self, crop_bgr, pointer_mask, geometry_reading, scale_start, scale_end, snap_out_of_range_pointer=True):
        reading = {
            "status": False,
            "backend": "geometry_hybrid",
            "message": "",
            "endNum": None,
            "resultNum": None,
            "endNum_float": None,
            "pointer_angle": None,
            "pointer_relative_angle": None,
            "progress_ratio": None,
            "p_geom": None,
            "delta_pred": None,
            "confidence": None,
            "cls_pred": None,
            "tip_info": None,
            "hybrid_model_path": str(self.model_path),
            "hybrid_model_loaded": self.model is not None,
            "geometry_source_backend": geometry_reading.get("backend"),
        }

        if self.model is None:
            reading["message"] = self.load_error or "hybrid model unavailable"
            return reading
        if not geometry_reading or not geometry_reading.get("status"):
            reading["message"] = "geometry baseline unavailable for hybrid backend"
            return reading

        prepared = self._prepare_inputs(crop_bgr, pointer_mask, geometry_reading)
        if prepared[0] is None:
            reading["message"] = prepared[2]
            return reading

        x, scalar, debug = prepared
        try:
            with torch.no_grad():
                delta, logits, confidence = self.model(x, scalar)
                delta_pred = float(delta.squeeze(0).cpu().item())
                cls_pred = int(torch.argmax(logits, dim=1).cpu().item())
                confidence_value = float(confidence.squeeze(0).cpu().item())

            p_geom = float(debug["p_geom"])
            raw_progress_ratio = p_geom + delta_pred
            circular_progress_ratio = circular_wrap_unit(raw_progress_ratio)
            progress_ratio = min(max(raw_progress_ratio, 0.0), 1.0)
            result_num = self._progress_to_result(progress_ratio, scale_start, scale_end, snap_out_of_range_pointer)
            if result_num is None:
                reading["message"] = "invalid scale range"
                return reading

            end_num_float = progress_ratio * 100.0
            end_num = int(round(min(max(end_num_float, 0.0), 100.0)))
            start_angle = geometry_reading.get("startAngle")
            dis_angle = geometry_reading.get("disAngle")
            pointer_angle = None
            pointer_relative_angle = None
            if start_angle is not None and dis_angle is not None:
                pointer_relative_angle = progress_ratio * float(dis_angle)
                pointer_angle = (float(start_angle) + pointer_relative_angle) % 360.0

            reading.update(
                status=True,
                message=f"几何+residual进度为{end_num_float:.2f}%，表盘读数是{result_num:.2f}",
                endNum=end_num,
                resultNum=result_num,
                endNum_float=end_num_float,
                pointer_angle=pointer_angle,
                pointer_relative_angle=pointer_relative_angle,
                progress_ratio=progress_ratio,
                progress_ratio_raw=raw_progress_ratio,
                progress_ratio_circular=circular_progress_ratio,
                progress_ratio_clamped=progress_ratio != raw_progress_ratio,
                p_geom=p_geom,
                delta_pred=delta_pred,
                confidence=confidence_value,
                cls_pred=cls_pred,
                tip_info=geometry_reading.get("tip_info"),
                startAngle=start_angle,
                disAngle=dis_angle,
            )
            return reading
        except Exception as exc:
            logger.warning(f"hybrid residual backend failed: {exc}")
            reading["message"] = f"hybrid residual backend failed: {exc}"
            return reading
