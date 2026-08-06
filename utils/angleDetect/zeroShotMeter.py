import sys,os

import argparse

floderPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(floderPath)

from pointerSeg.detectSeg import u2netpSeg
from yoloDetection.yoloDectect import targetDetectModel
from detect import meterFormer
from PIL import Image
import torch
import cv2
import math
import numpy as np
from loguru import logger

from geometry_baseline import estimate_pointer_tip, estimate_pointer_tip_v2
from residual_calibrator import apply_calibrator, load_calibrator, make_calibrator_row
from residual_hybrid_backend import ResidualHybridBackend, DEFAULT_MODEL_PATH as DEFAULT_HYBRID_MODEL_PATH

LEGACY_RESIDUAL_CALIBRATOR_PATH = os.path.abspath(os.path.join(
    floderPath,
    "..",
    "..",
    "improvement_artifacts_20260709",
    "models",
    "tabular_residual_learned_gate",
    "38sample_candidates",
    "20260710_120723_130359",
    "calibrator.joblib",
))
PAPER_RESIDUAL_CALIBRATOR_PATH = os.path.abspath(os.path.join(
    floderPath,
    "..",
    "..",
    "artifacts",
    "runs",
    "syncg_full",
    "calibrator.joblib",
))
# Keep the old public symbol for code that imported it directly.
DEFAULT_RESIDUAL_CALIBRATOR_PATH = LEGACY_RESIDUAL_CALIBRATOR_PATH
DEFAULT_RESIDUAL_CALIBRATORS = {
    "geometry_fusion_calibrated": LEGACY_RESIDUAL_CALIBRATOR_PATH,
    "compare": LEGACY_RESIDUAL_CALIBRATOR_PATH,
    "geometry_fusion_weighted_calibrated": PAPER_RESIDUAL_CALIBRATOR_PATH,
}

class meterZeroShot():
    def __init__(self, u2netWeights, yoloWeights, meterclipWeights, pointWeights, device):
        self.device = device

        self.pointerSeg = u2netpSeg(u2netWeights, self.device)
        logger.info(f"语义分割模型成功加载")
        self.meterDetect = targetDetectModel(yoloWeights)
        logger.info(f"表盘模型成功加载")
        self.vlmMeter = meterFormer(meterclipWeights, self.device)
        logger.info(f"多模态模型成功加载")
        self.pointerDetect = targetDetectModel(pointWeights)
        logger.info(f"起始点检测模型成功加载")

        self.label_text_list = list(map(str, range(101)))
        self.label_text = self.vlmMeter.model.texteconder.tokenize(self.label_text_list).to(device)

        #-O 表计识别模型初始 相对 默认极坐标系下的相对角度
        self.ornMeterNum = 155.25 
        #-O 表计识别模型每个刻度所占比的角度大小 
        self.oneTempAngle = 360/101
        self.last_error_code = None
        self.last_error_message = None
        #-O 最近一次图像矫正的中间结果（供报告/调试读取，生产路径忽略）
        self._correction_debug = {}
        self._last_reading_details = {}
        self._last_training_artifacts = {}
        self._residual_calibrator_cache = {}
        self._residual_hybrid_cache = {}

    def _processImg(self):
        #-O 图像预处理部分，等着添加呢
        pass

    """
        表计检测
        输入：1个cv:mat类型图片，表盘的最小值与最大值（最小值，最大值默认为0和1.6）
        返回：endNum：代表归一化的旋转角度
             resultNum：代表最终表的读数
             segPointer：指针掩码
             corpImg：表盘图像
    """   
    def Inference(
            self,
            img,
            scaleStart=0,
            scaleEnd=1.6,
            confidence=None,
            use_origin_when_no_meter=False,
            start_end_distance_threshold=None,
            start_end_position="start_left_end_right",
            snap_out_of_range_pointer=True,
            correction_mode="ransacFun",
            visualization_mode="standard",
            stretch_x_ratio=1.0,
            stretch_y_ratio=1.0,
            reading_offset=0.0,
            default_start_angle=45.0,
            default_range_angle=270.0,
            validate_mask_line=True,
            mask_center_threshold_ratio=0.10,
            reading_backend="transformer",
            geometry_fallback_to_transformer=False,
            residual_calibrator_path=None,
            residual_hybrid_model_path=None,
            residual_hybrid_max_abs_delta=0.05,
            reference_conditioned_backend=None,
    ):
        self._clear_last_error()
        #-O 最近一次推理的指针射线坐标（结果图坐标系，供报告/调试显式绘制）
        self._last_pointer_ray = None
        self._last_reading_details = {}
        self._last_training_artifacts = {}
        #-O 1.得到裁剪图像
        all_confidences, all_boxes, all_crops, _, best_idx = self.meterDetect.image_crop(
            img,
            confidence=confidence,
            use_origin_when_no_meter=use_origin_when_no_meter
        )
        if best_idx is None or not all_crops:
            self._set_last_error("meter_not_found", "未检测到表盘")
            logger.info(f"未检测到表盘")
            return None, None, None, None, None
        
        corpImg = all_crops[best_idx]
        meter_confidence = (
            float(all_confidences[best_idx])
            if best_idx < len(all_confidences)
            else None
        )
        meter_bbox = (
            np.asarray(all_boxes[best_idx], dtype=np.float32).copy()
            if best_idx < len(all_boxes)
            else None
        )
        #-O 图像矫正（技术尚不成熟，不推荐使用）
        origin_cropImg = corpImg.copy()
        corpImg = self._correct_meter_image(
            corpImg,
            correction_mode,
            confidence=confidence,
            start_end_position=start_end_position,
            stretch_x_ratio=stretch_x_ratio,
            stretch_y_ratio=stretch_y_ratio,
        )
        #-O 裁剪图像
        corpImgDis = corpImg.copy()

        corpImg = cv2.cvtColor(corpImg, cv2.COLOR_BGR2RGB)

        #-O 2.得到分割图像
        #-O 目前即使分割效果不佳，但模型依然有很强的识别能力（具体原因不清楚，这可能就是黑盒子的魅力吧）
        corpImg  = Image.fromarray(corpImg)
        segImg = self.pointerSeg.Inference(corpImg)
        #-O 返还最大联通域部分
        segImg = self._find_largest_component(segImg)
        #-O 指针掩码
        segImgDis = segImg.copy()
        if len(segImgDis.shape) == 2:
            segImgDis = cv2.cvtColor(segImgDis, cv2.COLOR_GRAY2BGR)
        if segImgDis.shape[:2] != corpImgDis.shape[:2]:
            segImgDis = cv2.resize(
                segImgDis,
                (corpImgDis.shape[1], corpImgDis.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        requested_backend = self._normalize_reading_backend(reading_backend)
        independent_direction_backends = {
            "reference_conditioned_final",
            "probabilistic_vector",
        }
        need_transformer = requested_backend in {
            "transformer",
            "compare",
            "reference_conditioned_final",
        } or geometry_fallback_to_transformer
        resolved_residual_calibrator_path = self._resolve_residual_calibrator_path(
            residual_calibrator_path,
            requested_backend,
        )

        #-O 3.transformer模型输入
        pointerImg = Image.fromarray(segImg).convert('L')
        meterImg = corpImg.convert('L')
        # -O 4.坐标转换后处理
        corpImg = np.array(corpImg)
        height, width = corpImg.shape[0:2]
        centerX = width//2
        centerY = height//2
        imgCenter = (centerX, centerY)

        mask_line_valid = self._is_pointer_mask_line_valid(
                segImgDis,
                imgCenter,
                mask_center_threshold_ratio,
        )
        mask_validation_failed = bool(validate_mask_line and not mask_line_valid)
        if (
            mask_validation_failed
            and requested_backend not in independent_direction_backends
        ):
            self._set_last_error("pointer_not_found", "无法找到指针")
            logger.info("指针掩码主轴未经过表盘中心附近，判定无法找到指针")
            return None, None, segImgDis, None, origin_cropImg
        if mask_validation_failed:
            logger.info(
                "指针掩码主轴未经过表盘中心附近；审计后端保留独立概率方向分支"
            )

        transformer_endNum = None
        if need_transformer and not mask_validation_failed:
            preImg ,_ = self.vlmMeter.Inference(pointerImg, meterImg, self.label_text)
            imgIndex = preImg.argmax(dim=-1).detach().cpu().numpy().squeeze()
            transformer_endNum = self._apply_reading_offset(self.label_text_list[imgIndex], reading_offset)

        #-O 这下面的每一步都是时间的沉淀与投入
        #-O 这里的目标检测会把起点和终点识别为同一个点（待解决）
        #-O 这里有点有趣，因为数据集问题，所以这里实际用灰度图，识别效果更佳(先拿灰度试把，实际结果还要看模型的效果)
        corpImg = cv2.cvtColor(corpImg, cv2.COLOR_BGR2GRAY)
        corpImg = np.stack((corpImg,) * 3, axis=-1)

        # pointerCenter, _  = self.pointerDetect.center_find(corpImg, classId=0)
        center_end, _  = self.pointerDetect.center_find(corpImg, classId=1, confidence=confidence)
        center_start, _  = self.pointerDetect.center_find(corpImg, classId=2, confidence=confidence)
        center_start, center_end = self._normalize_start_end_points(
            center_start,
            center_end,
            width,
            start_end_distance_threshold,
            start_end_position,
        )

        draw_debug_pointer = self._is_debug_visualization(visualization_mode)

        if center_start is not None and center_end is None:
            logger.info(f"检测到起点，未检测到终点")
            startAngle = self.calculate_angle(imgCenter, center_start)
            center_start = tuple(map(int, center_start))
            endAngle = (startAngle + float(default_range_angle)) % 360
            meterNum = (self.ornMeterNum - startAngle) % 360
            disAngle = float(default_range_angle)
            branch_name = "start_only"
        elif center_start is None and center_end is not None:
            logger.info(f"未检测到起点，检测到终点")
            endAngle = self.calculate_angle(imgCenter, center_end)
            center_end = tuple(map(int, center_end))
            startAngle = (endAngle - float(default_range_angle)) % 360
            meterNum = (self.ornMeterNum - startAngle) % 360
            disAngle = float(default_range_angle)
            branch_name = "end_only"
        elif center_start is None and center_end is None:
            logger.info(f"起点终点均未被检测到")
            startAngle = float(default_start_angle)
            endAngle = (startAngle + float(default_range_angle)) % 360
            meterNum = (self.ornMeterNum - startAngle) % 360
            disAngle = float(default_range_angle)
            branch_name = "default_start_end"
        else:
            logger.info(f"起点与终点检测成功")
            center_end = tuple(map(int, center_end))
            center_start = tuple(map(int, center_start))
            logger.info(f'{center_end}->{center_start}')
            startAngle = self.calculate_angle(imgCenter, center_start)
            meterNum = (self.ornMeterNum - startAngle) % 360
            endAngle = self.calculate_angle(imgCenter, center_end)
            disAngle = (endAngle - startAngle) % 360
            branch_name = "start_and_end"

        transformer_reading = {
            "status": False,
            "backend": "transformer",
            "message": "transformer backend skipped",
            "endNum": None,
            "resultNum": None,
            "endNum_float": None,
            "pointer_angle": None,
        }
        if need_transformer and transformer_endNum is not None:
            transformer_result = self._calculate_meter_result(
                transformer_endNum,
                meterNum,
                disAngle,
                scaleStart,
                scaleEnd,
                snap_out_of_range_pointer,
            )
            transformer_reading = {
                "status": True,
                "backend": "transformer",
                "message": f"旋转角度为{transformer_endNum}，表盘读数是{transformer_result:.2f}",
                "endNum": transformer_endNum,
                "resultNum": transformer_result,
                "endNum_float": float(transformer_endNum),
                "pointer_angle": self._calculate_pointer_angle(transformer_endNum, meterNum),
            }

        geometry_legacy_reading = self._build_geometry_reading(
            segImgDis,
            imgCenter,
            meterNum,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer,
            mask_center_threshold_ratio,
        )
        geometry_direct_reading = self._build_geometry_direct_reading(
            segImgDis,
            imgCenter,
            startAngle,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer,
            mask_center_threshold_ratio,
        )
        geometry_direct_v2_reading = self._build_geometry_direct_v2_reading(
            segImgDis,
            imgCenter,
            startAngle,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer,
            mask_center_threshold_ratio,
        )
        geometry_fusion_reading = self._build_geometry_fusion_reading(
            geometry_direct_reading,
            geometry_direct_v2_reading,
        )
        geometry_fusion_weighted_reading = self._build_geometry_weighted_fusion_reading(
            geometry_direct_reading,
            geometry_direct_v2_reading,
        )
        if mask_validation_failed:
            mask_failure_message = (
                "pointer mask failed the frozen center-line validation"
            )
            for failed_reading in (
                transformer_reading,
                geometry_legacy_reading,
                geometry_direct_reading,
                geometry_direct_v2_reading,
                geometry_fusion_reading,
                geometry_fusion_weighted_reading,
            ):
                failed_reading.update(
                    status=False,
                    message=mask_failure_message,
                    resultNum=None,
                    progress_ratio=None,
                )
        training_artifacts = {
            "corrected_crop_bgr": corpImgDis.copy() if corpImgDis is not None else None,
            "pointer_mask": segImgDis.copy() if segImgDis is not None else None,
            "segmentation_summary": dict(
                getattr(self.pointerSeg, "last_probability_summary", {}) or {}
            ),
            "imgCenter": tuple(map(int, imgCenter)),
            "center_start": tuple(map(int, center_start)) if center_start is not None else None,
            "center_end": tuple(map(int, center_end)) if center_end is not None else None,
            "startAngle": float(startAngle),
            "endAngle": float(endAngle),
            "disAngle": float(disAngle),
            "branch": branch_name,
            "meter_bbox": meter_bbox,
            "meter_confidence": meter_confidence,
            "mask_line_valid": bool(mask_line_valid),
            "mask_validation_failed": bool(mask_validation_failed),
        }
        mean_calibrator_path = (
            resolved_residual_calibrator_path
            if requested_backend in {"geometry_fusion_calibrated", "compare"}
            else None
        )
        weighted_calibrator_path = (
            resolved_residual_calibrator_path
            if requested_backend == "geometry_fusion_weighted_calibrated"
            else None
        )
        geometry_fusion_calibrated_reading = self._build_geometry_fusion_calibrated_reading(
            geometry_direct_reading,
            geometry_direct_v2_reading,
            geometry_fusion_reading,
            training_artifacts,
            scaleStart,
            scaleEnd,
            mean_calibrator_path,
        )
        geometry_fusion_weighted_calibrated_reading = self._build_geometry_fusion_calibrated_reading(
            geometry_direct_reading,
            geometry_direct_v2_reading,
            geometry_fusion_weighted_reading,
            training_artifacts,
            scaleStart,
            scaleEnd,
            weighted_calibrator_path,
            backend_name="geometry_fusion_weighted_calibrated",
        )
        if requested_backend in {"geometry_hybrid", "geometry_hybrid_gate"}:
            geometry_hybrid_reading = self._build_geometry_hybrid_reading(
                geometry_direct_reading,
                training_artifacts,
                scaleStart,
                scaleEnd,
                residual_hybrid_model_path,
                snap_out_of_range_pointer=snap_out_of_range_pointer,
                max_abs_delta=residual_hybrid_max_abs_delta,
            )
        else:
            geometry_hybrid_reading = {
                "status": False,
                "backend": "geometry_hybrid",
                "message": "hybrid backend was not requested",
            }
        reference_conditioned_final_reading = {
            "status": False,
            "backend": "reference_conditioned_final",
            "message": "reference-conditioned backend was not requested",
        }
        probabilistic_vector_reading = {
            "status": False,
            "backend": "probabilistic_vector",
            "message": "probabilistic-vector backend was not requested",
        }
        if requested_backend == "reference_conditioned_final":
            if reference_conditioned_backend is None:
                reference_conditioned_final_reading["message"] = (
                    "reference-conditioned production artifacts are not configured"
                )
            else:
                try:
                    try:
                        from .reference_conditioned_runtime import (
                            build_front_end_payload,
                            build_raw_payload,
                        )
                    except ImportError:
                        from reference_conditioned_runtime import (  # type: ignore[no-redef]
                            build_front_end_payload,
                            build_raw_payload,
                        )

                    raw_payload = build_raw_payload(
                        transformer_reading=transformer_reading,
                        geometry_reading=geometry_direct_reading,
                        geometry_v2_reading=geometry_direct_v2_reading,
                        mean_fusion_reading=geometry_fusion_reading,
                        weighted_fusion_reading=geometry_fusion_weighted_reading,
                        training_artifacts=training_artifacts,
                        scale_start=scaleStart,
                        scale_end=scaleEnd,
                    )
                    front_end_payload = build_front_end_payload(training_artifacts)
                    reference_conditioned_final_reading = (
                        reference_conditioned_backend.predict(
                            image_bgr=img,
                            raw_row=raw_payload,
                            front_end=front_end_payload,
                        )
                    )
                except Exception as exc:
                    logger.exception(
                        "reference-conditioned final backend inference failed"
                    )
                    reference_conditioned_final_reading = {
                        "status": False,
                        "backend": "reference_conditioned_final",
                        "message": (
                            "reference-conditioned final backend failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "error_code": "reference_conditioned_backend_exception",
                    }

        if requested_backend == "probabilistic_vector":
            probabilistic_vector_reading = (
                self._run_probabilistic_vector_backend(
                    reference_conditioned_backend=reference_conditioned_backend,
                    image_bgr=img,
                    training_artifacts=training_artifacts,
                    scale_start=scaleStart,
                    scale_end=scaleEnd,
                )
            )

        if requested_backend == "reference_conditioned_final":
            selected_reading = dict(reference_conditioned_final_reading)
        elif requested_backend == "probabilistic_vector":
            selected_reading = dict(probabilistic_vector_reading)
        else:
            selected_reading = self._select_reading_output(
                reading_backend=requested_backend,
                transformer_reading=transformer_reading,
                geometry_reading=geometry_direct_reading,
                geometry_v2_reading=geometry_direct_v2_reading,
                geometry_legacy_reading=geometry_legacy_reading,
                geometry_fusion_weighted_reading=geometry_fusion_weighted_reading,
                geometry_fusion_calibrated_reading=geometry_fusion_calibrated_reading,
                geometry_fusion_weighted_calibrated_reading=geometry_fusion_weighted_calibrated_reading,
                geometry_hybrid_reading=geometry_hybrid_reading,
                geometry_fallback_to_transformer=geometry_fallback_to_transformer,
            )
        self._last_reading_details = {
            "requested_backend": requested_backend,
            "branch": branch_name,
            "transformer": transformer_reading,
            "geometry": geometry_legacy_reading,
            "geometry_direct": geometry_direct_reading,
            "geometry_direct_v2": geometry_direct_v2_reading,
            "geometry_fusion": geometry_fusion_reading,
            "geometry_fusion_weighted": geometry_fusion_weighted_reading,
            "geometry_fusion_calibrated": geometry_fusion_calibrated_reading,
            "geometry_fusion_weighted_calibrated": geometry_fusion_weighted_calibrated_reading,
            "geometry_hybrid": geometry_hybrid_reading,
            "geometry_legacy": geometry_legacy_reading,
            "reference_conditioned_final": reference_conditioned_final_reading,
            "probabilistic_vector": probabilistic_vector_reading,
            "selected": selected_reading,
            "selected_backend": selected_reading.get("backend"),
        }
        self._last_training_artifacts = training_artifacts


        self._draw_reference_visuals(corpImgDis, imgCenter, center_start, center_end, startAngle, endAngle)
        display_endNum = selected_reading.get("endNum")
        if display_endNum is None and transformer_endNum is not None:
            display_endNum = transformer_endNum
        if selected_reading.get("backend") in {
            "geometry_direct",
            "geometry_direct_v2",
            "geometry_fusion",
            "geometry_fusion_weighted",
            "geometry_fusion_calibrated",
            "geometry_fusion_weighted_calibrated",
            "geometry_hybrid",
            "reference_conditioned_final",
            "probabilistic_vector",
        } and selected_reading.get("pointer_angle") is not None:
            model_pointer_ray = self._get_pointer_ray_from_image_angle(
                corpImgDis,
                imgCenter,
                selected_reading.get("pointer_angle"),
            )
        else:
            model_pointer_ray = self._get_model_pointer_ray(corpImgDis, imgCenter, startAngle, display_endNum, meterNum)
        self._last_pointer_ray = model_pointer_ray
        if draw_debug_pointer and model_pointer_ray is not None:
            self._draw_pointer_ray(corpImgDis, model_pointer_ray)
            self._draw_pointer_ray(segImgDis, model_pointer_ray)

        if not selected_reading.get("status"):
            self._set_last_error(
                selected_reading.get("error_code") or "reading_backend_failed",
                selected_reading.get("message") or "读数失败",
            )
            return None, None, segImgDis, None, origin_cropImg

        return selected_reading.get("endNum"), selected_reading.get("resultNum"), segImgDis, corpImgDis, origin_cropImg

    def _clear_last_error(self):
        self.last_error_code = None
        self.last_error_message = None

    def _set_last_error(self, code, message):
        self.last_error_code = code
        self.last_error_message = message

    @staticmethod
    def _apply_reading_offset(endNum, reading_offset):
        try:
            adjusted = (float(endNum) + float(reading_offset or 0.0)) % 101
        except (TypeError, ValueError):
            adjusted = float(endNum)
        if abs(adjusted - round(adjusted)) < 1e-6:
            return int(round(adjusted))
        return adjusted

    def _calculate_pointer_angle(self, endNum, meterNum):
        return (self.oneTempAngle * float(endNum) + meterNum) % 360

    @staticmethod
    def _normalize_reading_backend(reading_backend):
        requested = str(reading_backend or "transformer").strip().lower()
        aliases = {
            "direct_geometry": "geometry_direct",
            "pure_geometry": "geometry_direct",
            "geometry_pure": "geometry_direct",
            "geometry-direct": "geometry_direct",
            "geometrydirect": "geometry_direct",
            "geometry_direct2": "geometry_direct_v2",
            "geometry-direct-v2": "geometry_direct_v2",
            "geometrydirectv2": "geometry_direct_v2",
            "robust_geometry": "geometry_direct_v2",
            "geometry_fusion": "geometry_fusion",
            "geometry-fusion": "geometry_fusion",
            "fusion_geometry": "geometry_fusion",
            "fused_geometry": "geometry_fusion",
            "geometry_direct_fusion": "geometry_fusion",
            "geometry_fusion_weighted": "geometry_fusion_weighted",
            "geometry-fusion-weighted": "geometry_fusion_weighted",
            "quality_weighted_fusion": "geometry_fusion_weighted",
            "weighted_geometry_fusion": "geometry_fusion_weighted",
            "geometry_fusion_calibrated": "geometry_fusion_calibrated",
            "geometry-fusion-calibrated": "geometry_fusion_calibrated",
            "calibrated_geometry_fusion": "geometry_fusion_calibrated",
            "geometry_calibrated": "geometry_fusion_calibrated",
            "geometry_fusion_weighted_calibrated": "geometry_fusion_weighted_calibrated",
            "geometry-fusion-weighted-calibrated": "geometry_fusion_weighted_calibrated",
            "quality_weighted_fusion_calibrated": "geometry_fusion_weighted_calibrated",
            "weighted_geometry_calibrated": "geometry_fusion_weighted_calibrated",
            "geometry_hybrid": "geometry_hybrid",
            "geometry-hybrid": "geometry_hybrid",
            "geometry_residual_hybrid": "geometry_hybrid",
            "residual_hybrid": "geometry_hybrid",
            "learned_residual": "geometry_hybrid",
            "geometry_hybrid_gate": "geometry_hybrid_gate",
            "geometry-hybrid-gate": "geometry_hybrid_gate",
            "adaptive_hybrid": "geometry_hybrid_gate",
            "hybrid_gate": "geometry_hybrid_gate",
            "reference-conditioned-final": "reference_conditioned_final",
            "reference_conditioned": "reference_conditioned_final",
            "paper_final": "reference_conditioned_final",
            "paper-final": "reference_conditioned_final",
            "ours_final": "reference_conditioned_final",
            "ours-final": "reference_conditioned_final",
            "raw_probabilistic_vector": "probabilistic_vector",
            "raw-probabilistic-vector": "probabilistic_vector",
            "probabilistic-vector": "probabilistic_vector",
            "probabilistic_direction": "probabilistic_vector",
            "probabilistic-direction": "probabilistic_vector",
            "legacy_geometry": "geometry_legacy",
            "geometry_old": "geometry_legacy",
        }
        return aliases.get(requested, requested)

    @staticmethod
    def _finite_payload_float(value):
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        return normalized if math.isfinite(normalized) else None

    @classmethod
    def _probabilistic_vector_uncertainty(cls, vector_payload):
        return {
            "angle_std_degrees": cls._finite_payload_float(
                vector_payload.get("angle_std_degrees")
            ),
            "angle_log_variance": cls._finite_payload_float(
                vector_payload.get("angle_log_variance")
            ),
            "angle_bin_entropy": cls._finite_payload_float(
                vector_payload.get("angle_bin_entropy")
            ),
            "angle_bin_resultant_length": cls._finite_payload_float(
                vector_payload.get("angle_bin_resultant_length")
            ),
            "pivot_peak": cls._finite_payload_float(
                vector_payload.get("pivot_peak")
            ),
            "pivot_spatial_entropy": cls._finite_payload_float(
                vector_payload.get("pivot_spatial_entropy")
            ),
            "pivot_top2_margin": cls._finite_payload_float(
                vector_payload.get("pivot_top2_margin")
            ),
            "direction_raw_norm": cls._finite_payload_float(
                vector_payload.get("direction_raw_norm")
            ),
            "coverage_claim": (
                "diagnostic_only; angular sigma is not a calibrated interval"
            ),
        }

    def _run_probabilistic_vector_backend(
            self,
            *,
            reference_conditioned_backend,
            image_bgr,
            training_artifacts,
            scale_start,
            scale_end,
    ):
        """Return the frozen direction expert verbatim, without calibration/routing."""

        if reference_conditioned_backend is None:
            return {
                "status": False,
                "backend": "probabilistic_vector",
                "error_code": "probabilistic_vector_artifacts_not_configured",
                "message": (
                    "probabilistic-vector production artifacts are not configured"
                ),
                "prediction": None,
                "progress": None,
                "direction": None,
                "resultNum": None,
                "calibration_applied": False,
                "router_applied": False,
            }
        try:
            try:
                from .reference_conditioned_runtime import build_front_end_payload
            except ImportError:
                from reference_conditioned_runtime import (  # type: ignore[no-redef]
                    build_front_end_payload,
                )

            front_end_payload = build_front_end_payload(training_artifacts)
            vector_payload = reference_conditioned_backend.predict_direction(
                image_bgr,
                front_end_payload,
                scale_start=float(scale_start),
                scale_end=float(scale_end),
            )
        except Exception as exc:
            logger.exception("probabilistic-vector backend inference failed")
            return {
                "status": False,
                "backend": "probabilistic_vector",
                "error_code": "probabilistic_vector_backend_exception",
                "message": (
                    "probabilistic-vector backend failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "prediction": None,
                "progress": None,
                "direction": None,
                "resultNum": None,
                "calibration_applied": False,
                "router_applied": False,
                "artifact_audit": dict(
                    getattr(reference_conditioned_backend, "audit", {}) or {}
                ),
            }

        reading = dict(vector_payload or {})
        reading.update(
            backend="probabilistic_vector",
            calibration_applied=False,
            router_applied=False,
            decision_policy="raw_probabilistic_vector",
            artifact_audit=dict(
                getattr(reference_conditioned_backend, "audit", {}) or {}
            ),
        )
        reading["uncertainty"] = self._probabilistic_vector_uncertainty(reading)
        for key, value in reading["uncertainty"].items():
            if key != "coverage_claim":
                reading[key] = value
        if reading.get("status") is not True:
            reading.setdefault(
                "error_code",
                "probabilistic_vector_inference_failed",
            )
            reading.setdefault(
                "message",
                "probabilistic direction inference failed",
            )
            reading["prediction"] = None
            reading["progress"] = None
            reading["direction"] = None
            reading["resultNum"] = None
            reading["endNum"] = None
            reading["endNum_float"] = None
            return reading

        try:
            prediction = float(reading["prediction"])
            progress = float(reading["progress"])
            pointer_angle = float(reading["pointer_angle"])
            direction = [float(value) for value in reading["direction"]]
            if (
                not math.isfinite(prediction)
                or not math.isfinite(progress)
                or not math.isfinite(pointer_angle)
                or len(direction) != 2
                or not all(math.isfinite(value) for value in direction)
            ):
                raise ValueError("non-finite or malformed vector values")
        except (KeyError, TypeError, ValueError) as exc:
            reading.update(
                status=False,
                error_code="invalid_probabilistic_vector_payload",
                message=f"probabilistic direction payload is invalid: {exc}",
                prediction=None,
                progress=None,
                direction=None,
                resultNum=None,
                endNum=None,
                endNum_float=None,
            )
            return reading

        reading.update(
            prediction=prediction,
            progress=progress,
            pointer_angle=pointer_angle,
            direction=direction,
            resultNum=prediction,
            progress_ratio=progress,
            endNum=None,
            endNum_float=None,
            message=(
                "raw probabilistic direction vector returned without "
                "calibration or routing"
            ),
        )
        return reading

    @staticmethod
    def _build_geometry_fusion_reading(geometry_reading, geometry_v2_reading):
        reading = {
            "status": False,
            "backend": "geometry_fusion",
            "message": "",
            "endNum": None,
            "resultNum": None,
            "endNum_float": None,
            "pointer_angle": None,
            "pointer_relative_angle": None,
            "progress_ratio": None,
            "fusion_strategy": "mean_valid_geometry_direct_v1_v2",
            "fusion_sources": [],
            "source_delta_result": None,
            "source_delta_progress": None,
        }

        sources = []
        for source in (geometry_reading, geometry_v2_reading):
            if not source or not source.get("status"):
                continue
            if source.get("resultNum") is None or source.get("progress_ratio") is None:
                continue
            sources.append(source)

        if not sources:
            reading["message"] = "geometry fusion has no valid source readings"
            return reading

        result_values = [float(s["resultNum"]) for s in sources]
        progress_values = [float(s["progress_ratio"]) for s in sources]
        end_num_values = [float(s.get("endNum_float", s.get("endNum", 0.0))) for s in sources]
        pointer_angles = [s.get("pointer_angle") for s in sources if s.get("pointer_angle") is not None]
        relative_angles = [s.get("pointer_relative_angle") for s in sources if s.get("pointer_relative_angle") is not None]

        def circular_mean_deg(values):
            if not values:
                return None
            radians = np.radians([float(v) for v in values])
            sin_mean = float(np.mean(np.sin(radians)))
            cos_mean = float(np.mean(np.cos(radians)))
            return float((np.degrees(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)

        result_num = float(np.mean(result_values))
        progress_ratio = float(np.mean(progress_values))
        end_num_float = float(np.mean(end_num_values))
        end_num = int(round(min(max(end_num_float, 0.0), 100.0)))

        reading.update(
            status=True,
            message=f"融合几何进度为{end_num_float:.2f}%，表盘读数是{result_num:.2f}",
            endNum=end_num,
            resultNum=result_num,
            endNum_float=end_num_float,
            pointer_angle=circular_mean_deg(pointer_angles),
            pointer_relative_angle=circular_mean_deg(relative_angles),
            progress_ratio=progress_ratio,
            fusion_sources=[s.get("backend") for s in sources],
            source_delta_result=(max(result_values) - min(result_values)) if len(result_values) > 1 else 0.0,
            source_delta_progress=(max(progress_values) - min(progress_values)) if len(progress_values) > 1 else 0.0,
        )
        return reading

    @staticmethod
    def _geometry_source_quality(source):
        """Return an inference-only reliability score for a geometry source."""
        if not source or not source.get("status"):
            return 0.0
        tip_info = source.get("tip_info") or {}
        confidence = tip_info.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and math.isfinite(confidence):
            return float(np.clip(confidence, 0.0, 1.0))

        center_distance = tip_info.get("center_distance")
        threshold = tip_info.get("threshold")
        point_count = tip_info.get("point_count")
        candidate_count = tip_info.get("candidate_count")
        try:
            axis_score = 1.0 - float(center_distance) / max(float(threshold), 1e-6)
        except (TypeError, ValueError):
            axis_score = 0.0
        try:
            support_ratio = float(candidate_count) / max(float(point_count), 1.0)
        except (TypeError, ValueError):
            support_ratio = 0.0
        return float(np.clip(0.7 * np.clip(axis_score, 0.0, 1.0) + 0.3 * np.clip(support_ratio, 0.0, 1.0), 0.0, 1.0))

    @classmethod
    def _build_geometry_weighted_fusion_reading(cls, geometry_reading, geometry_v2_reading):
        """Fuse valid geometry estimates using mask-derived reliability only."""
        reading = {
            "status": False,
            "backend": "geometry_fusion_weighted",
            "message": "",
            "endNum": None,
            "resultNum": None,
            "endNum_float": None,
            "pointer_angle": None,
            "pointer_relative_angle": None,
            "progress_ratio": None,
            "fusion_strategy": "quality_weighted_geometry_direct_v1_v2",
            "fusion_sources": [],
            "fusion_source_weights": {},
            "fusion_source_quality": {},
            "source_delta_result": None,
            "source_delta_progress": None,
        }

        sources = []
        for source in (geometry_reading, geometry_v2_reading):
            if not source or not source.get("status"):
                continue
            if source.get("resultNum") is None or source.get("progress_ratio") is None:
                continue
            sources.append(source)

        if not sources:
            reading["message"] = "weighted geometry fusion has no valid source readings"
            return reading

        qualities = np.asarray([max(0.05, cls._geometry_source_quality(source)) for source in sources], dtype=np.float64)
        weights = qualities / max(float(np.sum(qualities)), 1e-9)

        def weighted_value(key, fallback_key=None):
            values = []
            value_weights = []
            for source, weight in zip(sources, weights):
                value = source.get(key)
                if value is None and fallback_key is not None:
                    value = source.get(fallback_key)
                if value is None:
                    continue
                values.append(float(value))
                value_weights.append(float(weight))
            if not values:
                return None
            value_weights = np.asarray(value_weights, dtype=np.float64)
            value_weights /= max(float(np.sum(value_weights)), 1e-9)
            return float(np.average(np.asarray(values, dtype=np.float64), weights=value_weights))

        def weighted_circular_mean(key):
            values = []
            value_weights = []
            for source, weight in zip(sources, weights):
                value = source.get(key)
                if value is None:
                    continue
                values.append(float(value))
                value_weights.append(float(weight))
            if not values:
                return None
            radians = np.radians(values)
            value_weights = np.asarray(value_weights, dtype=np.float64)
            sin_sum = float(np.sum(np.sin(radians) * value_weights))
            cos_sum = float(np.sum(np.cos(radians) * value_weights))
            if math.hypot(sin_sum, cos_sum) < 1e-9:
                return float(values[0] % 360.0)
            return float((np.degrees(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0)

        result_values = [float(source["resultNum"]) for source in sources]
        progress_values = [float(source["progress_ratio"]) for source in sources]
        result_num = weighted_value("resultNum")
        progress_ratio = weighted_value("progress_ratio")
        end_num_float = weighted_value("endNum_float", "endNum")
        end_num = int(round(min(max(end_num_float, 0.0), 100.0)))

        source_names = [str(source.get("backend") or f"source_{index}") for index, source in enumerate(sources)]
        reading.update(
            status=True,
            message=f"质量加权几何进度为{end_num_float:.2f}%，表盘读数是{result_num:.2f}",
            endNum=end_num,
            resultNum=result_num,
            endNum_float=end_num_float,
            pointer_angle=weighted_circular_mean("pointer_angle"),
            pointer_relative_angle=weighted_value("pointer_relative_angle"),
            progress_ratio=progress_ratio,
            fusion_sources=source_names,
            fusion_source_weights={
                name: float(weight) for name, weight in zip(source_names, weights)
            },
            fusion_source_quality={
                name: float(quality) for name, quality in zip(source_names, qualities)
            },
            source_delta_result=(max(result_values) - min(result_values)) if len(result_values) > 1 else 0.0,
            source_delta_progress=(max(progress_values) - min(progress_values)) if len(progress_values) > 1 else 0.0,
        )
        return reading

    def _load_residual_calibrator(self, calibrator_path):
        if not calibrator_path:
            return None
        key = os.path.abspath(str(calibrator_path))
        if key not in self._residual_calibrator_cache:
            self._residual_calibrator_cache[key] = load_calibrator(key)
        return self._residual_calibrator_cache[key]

    def _resolve_residual_calibrator_path(self, calibrator_path, requested_backend):
        if calibrator_path == "":
            return None
        if calibrator_path is not None:
            return calibrator_path
        default_path = DEFAULT_RESIDUAL_CALIBRATORS.get(requested_backend)
        if default_path and os.path.exists(default_path):
            return default_path
        return None

    def _load_residual_hybrid_backend(self, model_path):
        path = os.path.abspath(str(model_path or DEFAULT_HYBRID_MODEL_PATH))
        if path not in self._residual_hybrid_cache:
            self._residual_hybrid_cache[path] = ResidualHybridBackend(path, device=self.device)
        return self._residual_hybrid_cache[path]

    def _build_geometry_fusion_calibrated_reading(
            self,
            geometry_reading,
            geometry_v2_reading,
            geometry_fusion_reading,
            training_artifacts,
            scaleStart,
            scaleEnd,
            residual_calibrator_path=None,
            backend_name="geometry_fusion_calibrated",
    ):
        reading = dict(geometry_fusion_reading or {})
        reading.update(
            status=False,
            backend=backend_name,
            message="residual calibrator path is not configured",
        )
        if not geometry_fusion_reading or not geometry_fusion_reading.get("status"):
            reading["message"] = "geometry fusion base reading is invalid"
            return reading
        if not residual_calibrator_path:
            return reading

        try:
            package = self._load_residual_calibrator(residual_calibrator_path)
            feature_row = make_calibrator_row(
                geometry_reading,
                geometry_v2_reading,
                geometry_fusion_reading,
                training_artifacts,
                corrected_crop_bgr=training_artifacts.get("corrected_crop_bgr"),
                scale_start=scaleStart,
                scale_end=scaleEnd,
            )
            calibrated = apply_calibrator(
                package,
                feature_row,
                geometry_fusion_reading,
                backend_name=backend_name,
            )
            if calibrated is None:
                reading["message"] = "residual calibrator returned no reading"
                return reading
            return calibrated
        except Exception as exc:
            logger.warning(f"geometry fusion residual calibration failed: {exc}")
            reading["message"] = f"geometry fusion residual calibration failed: {exc}"
            return reading

    def _build_geometry_hybrid_reading(
            self,
            geometry_reading,
            training_artifacts,
            scaleStart,
            scaleEnd,
            residual_hybrid_model_path=None,
            snap_out_of_range_pointer=True,
            max_abs_delta=0.05,
    ):
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
            "startAngle": training_artifacts.get("startAngle"),
            "disAngle": training_artifacts.get("disAngle"),
        }

        if not geometry_reading or not geometry_reading.get("status"):
            reading["message"] = "geometry base reading is invalid"
            return reading

        crop_bgr = training_artifacts.get("corrected_crop_bgr")
        pointer_mask = training_artifacts.get("pointer_mask")
        if crop_bgr is None or pointer_mask is None:
            reading["message"] = "hybrid backend missing crop or mask"
            return reading

        try:
            backend = self._load_residual_hybrid_backend(residual_hybrid_model_path)
        except Exception as exc:
            logger.warning(f"load hybrid backend failed: {exc}")
            reading["message"] = f"load hybrid backend failed: {exc}"
            return reading

        predicted = backend.predict(
            crop_bgr,
            pointer_mask,
            geometry_reading,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer=snap_out_of_range_pointer,
        )
        if not predicted.get("status"):
            reading.update(predicted)
            return reading
        if max_abs_delta is not None:
            try:
                max_abs_delta = abs(float(max_abs_delta))
                delta_pred = abs(float(predicted.get("delta_pred", 0.0)))
                if delta_pred > max_abs_delta:
                    reading.update(predicted)
                    reading.update(
                        status=False,
                        message=f"hybrid residual gate: abs(delta) {delta_pred:.4f} > {max_abs_delta:.4f}",
                        hybrid_gate="max_abs_delta",
                        hybrid_gate_threshold=max_abs_delta,
                    )
                    return reading
            except (TypeError, ValueError):
                pass
        return predicted
    def _build_geometry_reading(
            self,
            seg_mask,
            center,
            meterNum,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer=True,
            mask_center_threshold_ratio=0.10,
    ):
        reading = {
            "status": False,
            "backend": "geometry",
            "message": "",
            "endNum": None,
            "resultNum": None,
            "endNum_float": None,
            "pointer_angle": None,
            "tip_point": None,
            "tip_info": None,
        }

        tip_info = estimate_pointer_tip(seg_mask, center, mask_center_threshold_ratio)
        reading["tip_info"] = tip_info
        if not tip_info.get("status"):
            reading["message"] = tip_info.get("message") or "geometry tip estimation failed"
            return reading

        tip_point = tip_info.get("tip_point")
        if tip_point is None:
            reading["message"] = "geometry tip point is None"
            return reading

        abs_angle = self.calculate_angle(center, tip_point)
        # end_num 需要满足 _calculate_pointer_angle(end_num, meterNum) == (abs_angle - startAngle) % 360
        # 化简得：end_num = (abs_angle - ornMeterNum) % 360 / oneTempAngle
        end_num_float = ((abs_angle - self.ornMeterNum) % 360) / float(self.oneTempAngle)
        end_num = int(round(end_num_float)) % 101
        result_num = self._calculate_meter_result(
            end_num,
            meterNum,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer,
        )

        reading.update(
            status=True,
            message=f"旋转角度为{end_num}，表盘读数是{result_num:.2f}",
            endNum=end_num,
            resultNum=result_num,
            endNum_float=end_num_float,
            pointer_angle=abs_angle,
            tip_point=tip_point,
        )
        return reading

    def _build_geometry_direct_reading(
            self,
            seg_mask,
            center,
            startAngle,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer=True,
            mask_center_threshold_ratio=0.10,
    ):
        reading = {
            "status": False,
            "backend": "geometry_direct",
            "message": "",
            "endNum": None,
            "resultNum": None,
            "endNum_float": None,
            "pointer_angle": None,
            "pointer_relative_angle": None,
            "progress_ratio": None,
            "confidence": 0.0,
            "startAngle": startAngle,
            "disAngle": disAngle,
            "tip_point": None,
            "tip_info": None,
        }

        tip_info = estimate_pointer_tip(seg_mask, center, mask_center_threshold_ratio)
        reading["tip_info"] = tip_info
        if not tip_info.get("status"):
            reading["message"] = tip_info.get("message") or "geometry tip estimation failed"
            return reading

        tip_point = tip_info.get("tip_point")
        if tip_point is None:
            reading["message"] = "geometry tip point is None"
            return reading

        abs_angle = self.calculate_angle(center, tip_point)
        pointer_relative_angle = (abs_angle - float(startAngle)) % 360
        result_num, progress_ratio = self._calculate_meter_result_from_relative_angle(
            pointer_relative_angle,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer,
        )
        end_num_float = progress_ratio * 100.0
        end_num = int(round(min(max(end_num_float, 0.0), 100.0)))

        reading.update(
            status=True,
            message=f"几何进度为{end_num_float:.2f}%，表盘读数是{result_num:.2f}",
            endNum=end_num,
            resultNum=result_num,
            endNum_float=end_num_float,
            pointer_angle=abs_angle,
            pointer_relative_angle=pointer_relative_angle,
            progress_ratio=progress_ratio,
            confidence=float(tip_info.get("confidence") or 0.0),
            tip_point=tip_point,
        )
        return reading

    def _build_geometry_direct_v2_reading(
            self,
            seg_mask,
            center,
            startAngle,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer=True,
            mask_center_threshold_ratio=0.10,
    ):
        reading = {
            "status": False,
            "backend": "geometry_direct_v2",
            "message": "",
            "endNum": None,
            "resultNum": None,
            "endNum_float": None,
            "pointer_angle": None,
            "pointer_relative_angle": None,
            "progress_ratio": None,
            "confidence": 0.0,
            "startAngle": startAngle,
            "disAngle": disAngle,
            "tip_point": None,
            "tip_info": None,
        }

        tip_info = estimate_pointer_tip_v2(seg_mask, center, mask_center_threshold_ratio)
        reading["tip_info"] = tip_info
        if not tip_info.get("status"):
            reading["message"] = tip_info.get("message") or "geometry v2 tip estimation failed"
            return reading

        tip_point = tip_info.get("tip_point")
        if tip_point is None:
            reading["message"] = "geometry v2 tip point is None"
            return reading

        abs_angle = self.calculate_angle(center, tip_point)
        pointer_relative_angle = (abs_angle - float(startAngle)) % 360
        result_num, progress_ratio = self._calculate_meter_result_from_relative_angle(
            pointer_relative_angle,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer,
        )
        end_num_float = progress_ratio * 100.0
        end_num = int(round(min(max(end_num_float, 0.0), 100.0)))

        reading.update(
            status=True,
            message=f"稳健几何进度为{end_num_float:.2f}%，表盘读数是{result_num:.2f}",
            endNum=end_num,
            resultNum=result_num,
            endNum_float=end_num_float,
            pointer_angle=abs_angle,
            pointer_relative_angle=pointer_relative_angle,
            progress_ratio=progress_ratio,
            confidence=float(tip_info.get("confidence") or 0.0),
            tip_point=tip_point,
        )
        return reading

    def _select_reading_output(
            self,
            reading_backend,
            transformer_reading,
            geometry_reading,
            geometry_v2_reading=None,
            geometry_legacy_reading=None,
            geometry_fusion_weighted_reading=None,
            geometry_fusion_calibrated_reading=None,
            geometry_fusion_weighted_calibrated_reading=None,
            geometry_hybrid_reading=None,
            geometry_fallback_to_transformer=False,
    ):
        requested = self._normalize_reading_backend(reading_backend)
        transformer_ok = bool(transformer_reading.get("status"))
        geometry_ok = bool(geometry_reading.get("status"))
        geometry_v2 = geometry_v2_reading or geometry_reading
        geometry_v2_ok = bool(geometry_v2.get("status"))
        geometry_fusion_reading = self._build_geometry_fusion_reading(geometry_reading, geometry_v2)
        geometry_fusion_ok = bool(geometry_fusion_reading.get("status"))
        geometry_weighted = geometry_fusion_weighted_reading or self._build_geometry_weighted_fusion_reading(
            geometry_reading,
            geometry_v2,
        )
        geometry_weighted_ok = bool(geometry_weighted.get("status"))

        if requested == "geometry":
            legacy = geometry_legacy_reading or geometry_reading
            legacy_ok = bool(legacy.get("status"))
            if legacy_ok:
                selected = dict(legacy)
                selected["backend"] = "geometry"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry"
                return selected
            selected = dict(legacy)
            selected["backend"] = "geometry"
            selected["status"] = False
            selected["message"] = legacy.get("message") or "geometry backend failed"
            return selected

        if requested == "geometry_direct":
            if geometry_ok:
                selected = dict(geometry_reading)
                selected["backend"] = "geometry_direct"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_direct"
                return selected
            selected = dict(geometry_reading)
            selected["backend"] = "geometry_direct"
            selected["status"] = False
            selected["message"] = geometry_reading.get("message") or "geometry direct backend failed"
            return selected

        if requested == "geometry_direct_v2":
            if geometry_v2_ok:
                selected = dict(geometry_v2)
                selected["backend"] = "geometry_direct_v2"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_direct_v2"
                return selected
            selected = dict(geometry_v2)
            selected["backend"] = "geometry_direct_v2"
            selected["status"] = False
            selected["message"] = geometry_v2.get("message") or "geometry direct v2 backend failed"
            return selected

        if requested == "geometry_fusion":
            if geometry_fusion_ok:
                return dict(geometry_fusion_reading)
            if geometry_ok:
                selected = dict(geometry_reading)
                selected["backend"] = "geometry_direct"
                selected["fallback_from"] = "geometry_fusion"
                return selected
            if geometry_v2_ok:
                selected = dict(geometry_v2)
                selected["backend"] = "geometry_direct_v2"
                selected["fallback_from"] = "geometry_fusion"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_fusion"
                return selected
            selected = dict(geometry_fusion_reading)
            selected["status"] = False
            selected["message"] = geometry_fusion_reading.get("message") or "geometry fusion backend failed"
            return selected

        if requested == "geometry_fusion_weighted":
            if geometry_weighted_ok:
                return dict(geometry_weighted)
            if geometry_fusion_ok:
                selected = dict(geometry_fusion_reading)
                selected["fallback_from"] = "geometry_fusion_weighted"
                return selected
            if geometry_ok:
                selected = dict(geometry_reading)
                selected["backend"] = "geometry_direct"
                selected["fallback_from"] = "geometry_fusion_weighted"
                return selected
            if geometry_v2_ok:
                selected = dict(geometry_v2)
                selected["backend"] = "geometry_direct_v2"
                selected["fallback_from"] = "geometry_fusion_weighted"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_fusion_weighted"
                return selected
            selected = dict(geometry_weighted)
            selected["status"] = False
            selected["message"] = geometry_weighted.get("message") or "weighted geometry fusion backend failed"
            return selected

        if requested == "geometry_fusion_calibrated":
            calibrated = geometry_fusion_calibrated_reading or {}
            if calibrated.get("status"):
                return dict(calibrated)
            if geometry_fusion_ok:
                selected = dict(geometry_fusion_reading)
                selected["fallback_from"] = "geometry_fusion_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            if geometry_ok:
                selected = dict(geometry_reading)
                selected["backend"] = "geometry_direct"
                selected["fallback_from"] = "geometry_fusion_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            if geometry_v2_ok:
                selected = dict(geometry_v2)
                selected["backend"] = "geometry_direct_v2"
                selected["fallback_from"] = "geometry_fusion_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_fusion_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            selected = dict(calibrated)
            selected["status"] = False
            selected["message"] = calibrated.get("message") or "geometry fusion calibrated backend failed"
            return selected

        if requested == "geometry_fusion_weighted_calibrated":
            calibrated = geometry_fusion_weighted_calibrated_reading or {}
            if calibrated.get("status"):
                return dict(calibrated)
            if geometry_weighted_ok:
                selected = dict(geometry_weighted)
                selected["fallback_from"] = "geometry_fusion_weighted_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            if geometry_fusion_ok:
                selected = dict(geometry_fusion_reading)
                selected["fallback_from"] = "geometry_fusion_weighted_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            if geometry_ok:
                selected = dict(geometry_reading)
                selected["backend"] = "geometry_direct"
                selected["fallback_from"] = "geometry_fusion_weighted_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            if geometry_v2_ok:
                selected = dict(geometry_v2)
                selected["backend"] = "geometry_direct_v2"
                selected["fallback_from"] = "geometry_fusion_weighted_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_fusion_weighted_calibrated"
                selected["calibration_status"] = calibrated.get("message") or "calibration unavailable"
                return selected
            selected = dict(calibrated)
            selected["status"] = False
            selected["message"] = calibrated.get("message") or "weighted geometry fusion calibrated backend failed"
            return selected

        if requested == "geometry_hybrid":
            hybrid = geometry_hybrid_reading or {}
            if hybrid.get("status"):
                return dict(hybrid)
            if geometry_fusion_calibrated_reading and geometry_fusion_calibrated_reading.get("status"):
                selected = dict(geometry_fusion_calibrated_reading)
                selected["backend"] = "geometry_fusion_calibrated"
                selected["fallback_from"] = "geometry_hybrid"
                return selected
            if geometry_fusion_ok:
                selected = dict(geometry_fusion_reading)
                selected["backend"] = "geometry_fusion"
                selected["fallback_from"] = "geometry_hybrid"
                return selected
            if geometry_ok:
                selected = dict(geometry_reading)
                selected["backend"] = "geometry_direct"
                selected["fallback_from"] = "geometry_hybrid"
                return selected
            if geometry_v2_ok:
                selected = dict(geometry_v2)
                selected["backend"] = "geometry_direct_v2"
                selected["fallback_from"] = "geometry_hybrid"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_hybrid"
                return selected
            selected = dict(hybrid)
            selected["status"] = False
            selected["message"] = hybrid.get("message") or "geometry hybrid backend failed"
            return selected

        if requested == "geometry_hybrid_gate":
            hybrid = geometry_hybrid_reading or {}
            gate_low = 45.14
            gate_high = 52.09
            gate_reason = "pointer_angle_out_of_gate"
            hybrid_angle = hybrid.get("pointer_angle")
            hybrid_ok = bool(hybrid.get("status"))
            in_gate = hybrid_ok and hybrid_angle is not None and gate_low <= float(hybrid_angle) <= gate_high
            if in_gate:
                selected = dict(hybrid)
                selected["backend"] = "geometry_hybrid"
                selected["hybrid_gate"] = "pointer_angle_window"
                selected["hybrid_gate_threshold"] = [gate_low, gate_high]
                return selected

            if geometry_fusion_calibrated_reading and geometry_fusion_calibrated_reading.get("status"):
                selected = dict(geometry_fusion_calibrated_reading)
                selected["backend"] = "geometry_fusion_calibrated"
                selected["fallback_from"] = "geometry_hybrid_gate"
                selected["hybrid_gate"] = gate_reason
                selected["hybrid_gate_threshold"] = [gate_low, gate_high]
                selected["hybrid_gate_hybrid_pointer_angle"] = hybrid_angle
                return selected
            if geometry_fusion_ok:
                selected = dict(geometry_fusion_reading)
                selected["backend"] = "geometry_fusion"
                selected["fallback_from"] = "geometry_hybrid_gate"
                selected["hybrid_gate"] = gate_reason
                selected["hybrid_gate_threshold"] = [gate_low, gate_high]
                selected["hybrid_gate_hybrid_pointer_angle"] = hybrid_angle
                return selected
            if geometry_ok:
                selected = dict(geometry_reading)
                selected["backend"] = "geometry_direct"
                selected["fallback_from"] = "geometry_hybrid_gate"
                selected["hybrid_gate"] = gate_reason
                selected["hybrid_gate_threshold"] = [gate_low, gate_high]
                selected["hybrid_gate_hybrid_pointer_angle"] = hybrid_angle
                return selected
            if geometry_v2_ok:
                selected = dict(geometry_v2)
                selected["backend"] = "geometry_direct_v2"
                selected["fallback_from"] = "geometry_hybrid_gate"
                selected["hybrid_gate"] = gate_reason
                selected["hybrid_gate_threshold"] = [gate_low, gate_high]
                selected["hybrid_gate_hybrid_pointer_angle"] = hybrid_angle
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_hybrid_gate"
                selected["hybrid_gate"] = gate_reason
                selected["hybrid_gate_threshold"] = [gate_low, gate_high]
                selected["hybrid_gate_hybrid_pointer_angle"] = hybrid_angle
                return selected
            selected = dict(hybrid)
            selected["status"] = False
            selected["message"] = hybrid.get("message") or "geometry hybrid gate backend failed"
            selected["hybrid_gate"] = gate_reason
            selected["hybrid_gate_threshold"] = [gate_low, gate_high]
            return selected

        if requested == "geometry_legacy":
            legacy = geometry_legacy_reading or geometry_reading
            if legacy.get("status"):
                selected = dict(legacy)
                selected["backend"] = "geometry_legacy"
                return selected
            if geometry_fallback_to_transformer and transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                selected["fallback_from"] = "geometry_legacy"
                return selected
            selected = dict(legacy)
            selected["backend"] = "geometry_legacy"
            selected["status"] = False
            selected["message"] = legacy.get("message") or "geometry legacy backend failed"
            return selected

        if requested == "compare":
            if transformer_ok:
                selected = dict(transformer_reading)
                selected["backend"] = "transformer"
                return selected
            legacy = geometry_legacy_reading or geometry_reading
            if legacy.get("status"):
                selected = dict(legacy)
                selected["backend"] = "geometry"
                return selected
            selected = dict(transformer_reading)
            selected["backend"] = "transformer"
            selected["status"] = False
            selected["message"] = transformer_reading.get("message") or legacy.get("message") or geometry_reading.get("message") or "all backends failed"
            return selected

        selected = dict(transformer_reading)
        selected["backend"] = "transformer"
        return selected

    @staticmethod
    def _is_debug_visualization(visualization_mode):
        if visualization_mode is None:
            return False
        return str(visualization_mode).strip().lower() in {
            "debug",
            "quick",
            "quick_browse",
            "browse",
            "full",
        }

    @staticmethod
    def _normalize_correction_mode(correction_mode):
        if correction_mode is None:
            return "off"

        mode = str(correction_mode).strip().lower()
        off_values = {"", "off", "close", "closed", "none", "false", "0", "disable", "disabled"}
        ransac_values = {"ransac", "ransacfun", "ransac_fun"}
        backup_values = {"backup", "ransacfunbackup", "ransacfun_backup", "ransac_backup"}
        square_values = {"square", "resize_square", "force_square"}
        stretch_values = {"stretch", "scale", "xy_stretch", "resize"}
        ellipse_values = {"ellipse", "ellipse2", "auto", "auto_ellipse", "dial", "circle", "deskew"}

        if mode in off_values:
            return "off"
        if mode in ransac_values:
            return "ransacFun"
        if mode in backup_values:
            return "ransacFunbackup"
        if mode in square_values:
            return "square"
        if mode in stretch_values:
            return "stretch"
        if mode in ellipse_values:
            return "ellipse"

        logger.warning(f"未知图像校正模式:{correction_mode}，已按关闭处理")
        return "off"

    def _correct_meter_image(
            self,
            corpImg,
            correction_mode,
            confidence=None,
            start_end_position="start_left_end_right",
            stretch_x_ratio=1.0,
            stretch_y_ratio=1.0
    ):
        mode = self._normalize_correction_mode(correction_mode)
        self._correction_debug = {"mode": mode, "applied": False}
        if mode == "off":
            logger.info("图像校正已关闭")
            return corpImg

        try:
            if mode == "square":
                corrected = self._resize_to_square(corpImg)
            elif mode == "stretch":
                corrected = self._stretch_image(corpImg, stretch_x_ratio, stretch_y_ratio)
            elif mode == "ellipse":
                corrected = self._ellipse_rectify(corpImg)
            elif mode == "ransacFun":
                corrected = self.pointerDetect.ransacFun(
                    corpImg,
                    confidence=confidence,
                    start_end_position=start_end_position,
                )
            else:
                corrected = self.pointerDetect.ransacFun_backup(
                    corpImg,
                    confidence=confidence,
                )
            logger.info(f"图像校正已成功:{mode}")
            return corrected
        except Exception as e:
            logger.info(f"图像校正失败:{mode}, {e}")
            return corpImg

    @staticmethod
    def _resize_to_square(image):
        height, width = image.shape[:2]
        size = max(height, width)
        if size <= 0:
            return image
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _safe_positive_ratio(value, default=1.0):
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return default
        if ratio <= 0:
            return default
        return ratio

    def _stretch_image(self, image, stretch_x_ratio=1.0, stretch_y_ratio=1.0):
        height, width = image.shape[:2]
        ratio_x = self._safe_positive_ratio(stretch_x_ratio)
        ratio_y = self._safe_positive_ratio(stretch_y_ratio)
        new_width = max(1, int(round(width * ratio_x)))
        new_height = max(1, int(round(height * ratio_y)))
        if new_width == width and new_height == height:
            return image
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # ===================== 椭圆自动矫正（无需人工俯仰角） =====================
    #-O 表盘是正圆，拍歪后在图里是椭圆。锁定白色表盘面拟椭圆，仿射压回正圆，
    #-O 从而自动消除俯仰/偏航带来的前缩，不需要外部提供拍摄角度。
    def _fit_face_ellipse(self, crop_bgr):
        """锁定中央白色表盘面并拟合椭圆，返回 (ellipse, face_mask) 或 (None, mask)。"""
        h, w = crop_bgr.shape[:2]
        area_img = float(h * w)
        cx0, cy0 = w / 2.0, h / 2.0
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        _, S_, V_ = cv2.split(hsv)
        # 白色表盘面：高亮度 + 低饱和
        v_thr = np.percentile(V_, 55)
        mask = ((V_ >= v_thr) & (S_ <= 90)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)

        num, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
        if num < 2:
            return None, mask
        best, best_score = None, -1.0
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.10 * area_img:
                continue
            ccx, ccy = cents[i]
            centric = math.hypot(ccx - cx0, ccy - cy0) / math.hypot(cx0, cy0)
            if centric > 0.40:
                continue
            score = area * (1 - centric)
            if score > best_score:
                best_score, best = score, i
        if best is None:
            return None, mask
        comp = (lab == best).astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, comp
        c = max(cnts, key=cv2.contourArea)
        if len(c) < 20:
            return None, comp
        ellipse = cv2.fitEllipse(c)
        (ecx, ecy), (d1, d2), _ = ellipse
        a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0
        if b < 1:
            return None, comp
        ratio = b / a
        fill = math.pi * a * b / area_img
        centric = math.hypot(ecx - cx0, ecy - cy0) / math.hypot(cx0, cy0)
        # 质量校验：不太扁、占比合理、较居中；不过关回退
        if ratio < 0.45 or fill < 0.12 or fill > 1.0 or centric > 0.35:
            logger.info(f"表盘面椭圆质量不达标(ratio={ratio:.2f},fill={fill:.2f},centric={centric:.2f})，回退")
            return None, comp
        return ellipse, comp

    @staticmethod
    def _rectify_ellipse_affine(crop_bgr, ellipse):
        """把椭圆压成正圆（半径取长半轴）的仿射矫正，绕椭圆中心。"""
        (cx, cy), (d1, d2), ang = ellipse
        a1, a2 = d1 / 2.0, d2 / 2.0
        target = max(a1, a2)
        th = math.radians(ang)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        S = np.diag([target / a1, target / a2])
        A = R @ S @ R.T
        center = np.array([cx, cy])
        t = center - A @ center
        M = np.hstack([A, t.reshape(2, 1)]).astype(np.float32)
        h, w = crop_bgr.shape[:2]
        return cv2.warpAffine(crop_bgr, M, (w, h), borderValue=(0, 0, 0))

    def _ellipse_rectify(self, corpImg):
        """表盘面椭圆自动矫正主入口；拟合失败或表盘已够圆则原样返回。"""
        ellipse, face_mask = self._fit_face_ellipse(corpImg)
        dbg = {"mode": "ellipse", "applied": False, "face_mask": face_mask,
               "ellipse_overlay": None, "rectified": None, "ellipse": ellipse,
               "yolo_center": None, "hybrid_ellipse": None}
        if ellipse is None:
            logger.info("未拟合到表盘面椭圆，矫正回退原图")
            self._correction_debug = dbg
            return corpImg

        # 尝试用YOLO检测表盘中心点（classId=0）来替换拟合的椭圆中心
        try:
            yolo_center, _ = self.pointerDetect.center_find(corpImg, classId=0)
            if yolo_center is not None:
                # 用YOLO中心替换椭圆中心，保留拟合的长短轴和角度
                (ecx, ecy), (d1, d2), ang = ellipse
                h, w = corpImg.shape[:2]
                cx0, cy0 = w / 2.0, h / 2.0
                # 验证YOLO中心是否合理（不要离图像中心太远）
                yolo_dist = math.hypot(yolo_center[0] - cx0, yolo_center[1] - cy0)
                max_dist = math.hypot(cx0, cy0) * 0.5  # 最多偏离对角线一半
                if yolo_dist <= max_dist:
                    ellipse = (yolo_center, (d1, d2), ang)
                    dbg["yolo_center"] = yolo_center
                    dbg["hybrid_ellipse"] = ellipse
                    logger.info(f"使用YOLO检测中心{yolo_center}替换拟合中心({ecx:.1f},{ecy:.1f})")
                else:
                    logger.info(f"YOLO中心{yolo_center}偏离过大，使用拟合中心")
        except Exception as e:
            logger.info(f"YOLO中心检测失败: {e}，使用拟合椭圆中心")

        (_, _), (d1, d2), _ = ellipse
        ratio = min(d1, d2) / max(d1, d2)
        overlay = corpImg.copy()
        cv2.ellipse(overlay, ellipse, (0, 255, 0), 2)
        dbg["ellipse_overlay"] = overlay
        # 正表盘保护：已经够圆就不动，避免把好图扰坏
        if ratio >= 0.95:
            logger.info(f"表盘已接近正圆(ratio={ratio:.3f})，跳过矫正")
            dbg["rectified"] = corpImg.copy()
            self._correction_debug = dbg
            return corpImg
        rect = self._rectify_ellipse_affine(corpImg, ellipse)
        dbg["applied"] = True
        dbg["rectified"] = rect
        self._correction_debug = dbg
        logger.info(f"表盘面椭圆矫正完成(ratio={ratio:.3f})")
        return rect

    def _is_pointer_mask_line_valid(self, mask, center, threshold_ratio=0.10):
        if mask is None or center is None:
            return False

        if len(mask.shape) == 3:
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            gray_mask = mask

        points = cv2.findNonZero((gray_mask > 0).astype(np.uint8))
        if points is None or len(points) < 2:
            logger.info("指针掩码为空或点数不足")
            return False

        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        direction_norm = math.hypot(float(vx), float(vy))
        if direction_norm < 1e-6:
            logger.info("指针掩码主轴无效")
            return False

        cx, cy = center
        distance = abs(float(vx) * (float(cy) - float(y0)) - float(vy) * (float(cx) - float(x0))) / direction_norm
        height, width = gray_mask.shape[:2]
        try:
            ratio = float(threshold_ratio)
        except (TypeError, ValueError):
            ratio = 0.10
        threshold = min(height, width) * max(0.0, ratio)
        logger.info(f"指针掩码中心线距离:{distance:.2f}, 阈值:{threshold:.2f}")
        return distance <= threshold

    def _angle_to_point(self, center, angle_deg, length, angle=180):
        """
        calculate_angle 的反向换算：由角度和中心点得到图像坐标点。
        """
        theta = math.radians((angle_deg + angle) % 360)
        end_x = center[0] + math.sin(theta) * length
        end_y = center[1] - math.cos(theta) * length
        return int(round(end_x)), int(round(end_y))

    def _get_default_reference_point(self, image, center, angle_deg):
        height, width = image.shape[:2]
        length = max(12.0, min(height, width) * 0.35)
        return self._angle_to_point(center, angle_deg, length)

    def _draw_reference_visuals(
            self,
            image,
            center,
            center_start=None,
            center_end=None,
            startAngle=None,
            endAngle=None
    ):
        """
        绘制中心点、起点、终点以及中心点到起终点的参考线。

        当起点或终点缺失时，使用当前推理分支的默认角度反推一个默认点，
        并用空心圆标出，方便区分真实检测点和默认补点。
        """
        if image is None or center is None:
            return

        center = tuple(map(int, center))
        start_point = tuple(map(int, center_start)) if center_start is not None else None
        end_point = tuple(map(int, center_end)) if center_end is not None else None

        if start_point is None and startAngle is not None:
            start_point = self._get_default_reference_point(image, center, startAngle)
        if end_point is None and endAngle is not None:
            end_point = self._get_default_reference_point(image, center, endAngle)

        self._draw_center_to_reference_points(image, center, start_point, end_point)
        cv2.circle(image, center, 10, (0, 100, 0), -1)

        if start_point is not None:
            thickness = -1 if center_start is not None else 2
            cv2.circle(image, start_point, 10, (0, 0, 100), thickness)
        if end_point is not None:
            thickness = -1 if center_end is not None else 2
            cv2.circle(image, end_point, 10, (0, 0, 255), thickness)

    def _draw_center_to_reference_points(self, image, center, center_start=None, center_end=None):
        """
        绘制中心点到起点、终点的参考射线。
        """
        if image is None or center is None:
            return

        center = tuple(map(int, center))
        if center_start is not None:
            cv2.line(
                image,
                center,
                tuple(map(int, center_start)),
                (255, 0, 0),
                2,
                lineType=cv2.LINE_AA,
            )
        if center_end is not None:
            cv2.line(
                image,
                center,
                tuple(map(int, center_end)),
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA,
            )

    def _get_model_pointer_ray(self, image, center, startAngle, endNum, meterNum=None):
        """
        根据结果图坐标系计算模型认为的指针射线。
        """
        if image is None or center is None or endNum is None:
            return None

        if meterNum is None:
            meterNum = (self.ornMeterNum - startAngle) % 360

        pointer_angle = self._calculate_pointer_angle(endNum, meterNum)
        image_angle = (startAngle + pointer_angle) % 360
        height, width = image.shape[:2]
        ray_length = max(height, width)
        center = tuple(map(int, center))
        end_point = self._angle_to_point(center, image_angle, ray_length)
        logger.info(
            f"模型指针相对角度:{pointer_angle:.2f}, 图像射线角度:{image_angle:.2f}"
        )
        return center, end_point

    def _get_pointer_ray_from_image_angle(self, image, center, image_angle):
        if image is None or center is None or image_angle is None:
            return None

        height, width = image.shape[:2]
        ray_length = max(height, width)
        center = tuple(map(int, center))
        end_point = self._angle_to_point(center, float(image_angle), ray_length)
        logger.info(f"几何指针图像射线角度:{float(image_angle):.2f}")
        return center, end_point

    def _draw_pointer_ray(self, image, pointer_ray):
        """
        按同一组结果图射线坐标绘制指针方向。
        """
        if image is None or pointer_ray is None:
            return

        center, end_point = pointer_ray
        cv2.arrowedLine(
            image,
            center,
            end_point,
            (0, 255, 255),
            3,
            line_type=cv2.LINE_AA,
            tipLength=0.08,
        )

    def _calculate_meter_result(
            self,
            endNum,
            meterNum,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer=True
    ):
        """
        根据指针分类角度和表盘量程角度计算读数。

        当计算结果超过量程时，默认将指针吸附到更近的端点：
        靠近 start 归到 scaleStart，靠近 end 归到 scaleEnd。
        """
        if disAngle is None or abs(disAngle) < 1e-6:
            logger.info("量程角度无效，返回起始量程值")
            return scaleStart

        scale = scaleEnd - scaleStart
        pointer_angle = self._calculate_pointer_angle(endNum, meterNum)
        resultNum = pointer_angle / disAngle
        resultNum = resultNum * scale + scaleStart

        low_scale = min(scaleStart, scaleEnd)
        high_scale = max(scaleStart, scaleEnd)
        is_out_of_range = resultNum < low_scale or resultNum > high_scale

        if snap_out_of_range_pointer and is_out_of_range:
            distance_to_start = min(pointer_angle, 360 - pointer_angle)
            distance_to_end = abs(pointer_angle - disAngle)
            distance_to_end = min(distance_to_end, 360 - distance_to_end)

            if distance_to_start <= distance_to_end:
                logger.info(
                    f"指针计算结果超过量程({resultNum:.4f})，靠近起点，归到{scaleStart}"
                )
                return scaleStart

            logger.info(
                f"指针计算结果超过量程({resultNum:.4f})，靠近终点，归到{scaleEnd}"
            )
            return scaleEnd

        if resultNum < scaleStart:
            logger.info(
                f"指针计算结果过小({resultNum:.4f})，靠近起点，归到{scaleStart}"
            )
            return scaleStart

        if resultNum > scaleEnd:
            logger.info(
                f"指针计算结果过大({resultNum:.4f})，靠近终点，归到{scaleEnd}"
            )
            return scaleEnd

        return resultNum

    def _calculate_meter_result_from_relative_angle(
            self,
            pointer_relative_angle,
            disAngle,
            scaleStart,
            scaleEnd,
            snap_out_of_range_pointer=True,
    ):
        """
        Pure geometry reading conversion.

        pointer_relative_angle is already measured from the start reference point,
        so this path does not depend on transformer-specific endNum or ornMeterNum.
        """
        if disAngle is None or abs(float(disAngle)) < 1e-6:
            logger.info("量程角度无效，返回起始量程值")
            return scaleStart, 0.0

        pointer_relative_angle = float(pointer_relative_angle)
        disAngle = float(disAngle)
        scale = scaleEnd - scaleStart
        progress_ratio = pointer_relative_angle / disAngle
        resultNum = progress_ratio * scale + scaleStart

        low_scale = min(scaleStart, scaleEnd)
        high_scale = max(scaleStart, scaleEnd)
        is_out_of_range = resultNum < low_scale or resultNum > high_scale

        if snap_out_of_range_pointer and is_out_of_range:
            distance_to_start = min(pointer_relative_angle, 360 - pointer_relative_angle)
            distance_to_end = abs(pointer_relative_angle - disAngle)
            distance_to_end = min(distance_to_end, 360 - distance_to_end)

            if distance_to_start <= distance_to_end:
                logger.info(
                    f"几何指针计算结果超过量程({resultNum:.4f})，靠近起点，归到{scaleStart}"
                )
                return scaleStart, 0.0

            logger.info(
                f"几何指针计算结果超过量程({resultNum:.4f})，靠近终点，归到{scaleEnd}"
            )
            return scaleEnd, 1.0

        if resultNum < scaleStart:
            logger.info(
                f"几何指针计算结果过小({resultNum:.4f})，靠近起点，归到{scaleStart}"
            )
            return scaleStart, 0.0

        if resultNum > scaleEnd:
            logger.info(
                f"几何指针计算结果过大({resultNum:.4f})，靠近终点，归到{scaleEnd}"
            )
            return scaleEnd, 1.0

        return resultNum, progress_ratio

    def calculate_angle(self, C, P, angle=180):
        """
        angle为180
        opencv坐标系         y轴方向
            .--------------------->
            |
            |
            |
            |
            |
            |
            |
            v
        x轴方向

        计算点 P 相对于旋转中心 C 的旋转角度（从正X轴逆时针方向）
        :param C: 参考点 (cx, cy)
        :param P: 目标点 (px, py)
        :return: 角度（度数）
        """
        dx = P[0] - C[0]
        dy = P[1] - C[1]
        angle_rad = math.atan2(dx, -dy)
        angle_deg = math.degrees(angle_rad)
        angle_deg = (angle_deg - angle) % 360
        return angle_deg

    @staticmethod
    def _is_start_expected_on_left(start_end_position):
        if isinstance(start_end_position, str):
            position = start_end_position.strip().lower()
        else:
            position = "start_left_end_right"

        right_start_values = {
            "start_right_end_left",
            "right_left",
            "start_right",
            "end_left",
            "rl",
        }
        return position not in right_start_values

    def _normalize_start_end_points(
            self,
            center_start,
            center_end,
            image_width,
            distance_threshold=None,
            start_end_position="start_left_end_right"
    ):
        """
        规范起点/终点检测结果。

        默认认为 start 在画面左半边、end 在画面右半边；当配置为
        start_right_end_left 时反过来。两个点距离小于阈值时按同一个点处理，
        再根据该点落在左半边还是右半边决定保留为 start 或 end。
        """
        start_left = self._is_start_expected_on_left(start_end_position)
        if distance_threshold is None:
            distance_threshold = max(10.0, float(image_width) * 0.035)
        else:
            distance_threshold = max(0.0, float(distance_threshold))

        def point_on_start_side(point):
            if point is None:
                return False
            point_x = float(point[0])
            if start_left:
                return point_x <= image_width / 2.0
            return point_x >= image_width / 2.0

        def split_single_point(point, source_name):
            if point_on_start_side(point):
                logger.info(f"仅检测到{source_name}，根据左右半边判断按起点处理")
                return point, None
            logger.info(f"仅检测到{source_name}，根据左右半边判断按终点处理")
            return None, point

        if center_start is not None and center_end is not None:
            distance = float(np.linalg.norm(np.float32(center_start) - np.float32(center_end)))
            if distance <= distance_threshold:
                same_point = center_start
                if point_on_start_side(same_point):
                    logger.info(f"起点终点距离过近({distance:.2f} <= {distance_threshold:.2f})，按起点处理")
                    return same_point, None
                logger.info(f"起点终点距离过近({distance:.2f} <= {distance_threshold:.2f})，按终点处理")
                return None, same_point

            start_should_swap = not point_on_start_side(center_start)
            end_on_start_side = point_on_start_side(center_end)
            if start_should_swap and end_on_start_side:
                logger.info("起点终点左右位置与配置相反，已交换")
                return center_end, center_start

        elif center_start is not None:
            return split_single_point(center_start, "起点")

        elif center_end is not None:
            return split_single_point(center_end, "终点")

        return center_start, center_end

    #-O 返还最大的联通域（这段代码的问题后续可以通过增加分割模型的数据量来进行解决）    
    def _find_largest_component(self, mask):
        """
        找到二值图像中最大的连通域并返回其对应的二值图像
        
        参数:
        mask: 输入的二值图像（单通道，0-255）
        
        返回:
        output_mask: 只包含最大连通域的二进制图像
        """
        # 用OpenCV找连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        if num_labels < 2:  # 只有背景，没有连通域
            return np.zeros_like(mask)
        
        # 找最大连通域所对应的标签（排除背景0）
        max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        
        # 创建一个新的二值图像，只有最大连通域为255，其他为0
        output_mask = np.zeros_like(mask)
        output_mask[labels == max_label] = 255
        
        return output_mask


if __name__  == "__main__":
    parser = argparse.ArgumentParser(
        description="Run meterZeroShot directly on one local image."
    )
    parser.add_argument("image", help="input image path")
    parser.add_argument("--scale-start", type=float, default=0.0)
    parser.add_argument("--scale-end", type=float, default=1.6)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="print the result without opening OpenCV windows",
    )
    args = parser.parse_args()

    u2netWeights = os.path.join(floderPath, "pointerSeg", "resultSeg", "best.pt")
    yoloWeights = os.path.join(
        floderPath, "yoloDetection", "result", "yolo_findMeter.pt"
    )
    meterclipWeights = os.path.join(
        floderPath, "vitTranforms", "result", "best.pt"
    )
    pointWeights = os.path.join(
        floderPath, "yoloDetection", "result", "yolo_pointbest.pt"
    )

    img = cv2.imread(os.path.abspath(args.image))
    if img is None:
        parser.error(f"cannot decode image: {args.image}")
    ZeroShotM = meterZeroShot(
        u2netWeights,
        yoloWeights,
        meterclipWeights,
        pointWeights,
        torch.device(args.device),
    )
    endNum, resultNum, segPointer, corpImg, origin_crop = ZeroShotM.Inference(
        img,
        args.scale_start,
        args.scale_end,
    )
    print("endNum:", endNum, "resultNum:", resultNum)

    if not args.no_display and segPointer is not None and corpImg is not None:
        cv2.namedWindow("pointer_mask", cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow("meter_crop", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("pointer_mask", segPointer)
        cv2.imshow("meter_crop", corpImg)
        cv2.waitKey(0)










