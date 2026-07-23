import os.path
import threading

import torch
from loguru import logger
import base64
import cv2

from models.BaseInferModel import BaseInferModel

from utils.angleDetect.zeroShotMeter import meterZeroShot


class PointerMeterInferModel(BaseInferModel):
    """
    指针表生产推理适配器。

    负责加载表盘检测、指针分割、读数和起终参考模型，将 API 配置别名规范化，
    并把 ``meterZeroShot`` 的图像结果序列化为 HTTP 响应所需的 Base64 字符串。
    """

    def __init__(self):
        super().__init__()

    def init(self, *args, **kwargs):
        logger.info("初始化模型")
        self.ZeroShotM = None
        self._infer_lock = threading.Lock()
        self.load_model()
    
    @logger.catch(reraise=True)
    def load_model(self):
        logger.info("加载模型")

        filePath = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(os.path.dirname(filePath), "utils")
        default_u2net_weights = os.path.join(
            weights_path,
            "angleDetect",
            "pointerSeg",
            "resultSeg",
            "best.pt",
        )
        u2netWeights = os.environ.get(
            "POINTER_METER_SEGMENTATION_WEIGHTS",
            default_u2net_weights,
        )
        u2netWeights = os.path.abspath(
            os.path.expanduser(os.path.expandvars(u2netWeights))
        )
        if not os.path.isfile(u2netWeights):
            raise FileNotFoundError(
                "指针分割权重不存在: "
                f"{u2netWeights}（可通过 POINTER_METER_SEGMENTATION_WEIGHTS 配置）"
            )
        yoloWeights = os.path.join(
            weights_path,
            "angleDetect",
            "yoloDetection",
            "result",
            "yolo_findMeter.pt",
        )
        meterclipWeights = os.path.join(
            weights_path,
            "angleDetect",
            "vitTranforms",
            "result",
            "best.pt",
        )
        pointWeights = os.path.join(
            weights_path,
            "angleDetect",
            "yoloDetection",
            "result",
            "yolo_pointbest.pt",
        )

        device = os.environ.get("POINTER_METER_DEVICE", "cpu").strip() or "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"POINTER_METER_DEVICE={device!r}，但当前 PyTorch 无法使用 CUDA"
            )
        logger.info(f"指针分割权重: {u2netWeights}")
        logger.info(f"推理设备: {device}")
        self.ZeroShotM = meterZeroShot(u2netWeights, yoloWeights, meterclipWeights,
                                  pointWeights, device)

        logger.info("模型加载成功")

    def infer(self, *args, **kwargs):
        """
        模型推理总线。
        Args:
            args: 位置参数。
            kwargs: 关键字参数。
        Returns:
            result (json): 模型推理的结果。
        """
        if self.ZeroShotM is None:
            logger.error("模型未加载成功")
            self.load_model()
        image_data = kwargs.get('image', None)
        config = kwargs.get('config', {}) or {}
        if image_data is None:
            return {"status":False, "message": "image_data is None", "result": [],
                    "result_image": None}
        infer_mode = config.get('infer_mode', 'infer')
        if infer_mode == 'infer':
            with self._infer_lock:
                result = self.infer_image_data_one_pic(image_data,config)
        else:
            return {"status": False,
                    "message": "config:infer_mode is not exist, please choose in [infer]",
                    "result": [],
                    "result_image": None}
        logger.info("推理成功")
        return result

    @staticmethod
    def _get_optional_float_config(config, *keys):
        for key in keys:
            value = config.get(key, None)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning(f"config:{key}={value} 不是有效数值，已忽略该配置")
                return None
        return None

    @classmethod
    def _get_float_config(cls, config, *keys, default=0.0):
        value = cls._get_optional_float_config(config, *keys)
        if value is None:
            return default
        return value

    @staticmethod
    def _get_bool_config(config, *keys, default=False):
        for key in keys:
            if key not in config:
                continue
            value = config[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "y", "on")
            return bool(value)
        return default

    def infer_image_data_one_pic(self, image, config):
        """
        只对一张图片进行推理，默认检测逻辑
        """
        logger.info("开始检测")
        config = config or {}
        confidence = self._get_optional_float_config(config, "confidence", "conf")
        use_origin_when_no_meter = self._get_bool_config(
            config,
            "use_origin_when_no_meter",
            "continue_infer_without_meter",
            "infer_origin_when_no_meter",
            default=False
        )
        start_end_distance_threshold = self._get_optional_float_config(
            config,
            "start_end_distance_threshold",
            "same_point_distance_threshold",
            "point_same_distance_threshold",
        )
        start_end_position = config.get(
            "start_end_position",
            config.get("start_end_layout", "start_left_end_right")
        )
        snap_out_of_range_pointer = self._get_bool_config(
            config,
            "snap_out_of_range_pointer",
            "clamp_out_of_range_pointer",
            "snap_pointer_when_out_of_range",
            default=True
        )
        correction_mode = config.get(
            "correction_mode",
            config.get("image_correction", config.get("correction_algorithm", "ransacFun"))
        )
        stretch_x_ratio = self._get_float_config(
            config,
            "stretch_x_ratio",
            "horizontal_stretch_ratio",
            default=1.0
        )
        stretch_y_ratio = self._get_float_config(
            config,
            "stretch_y_ratio",
            "vertical_stretch_ratio",
            default=1.0
        )
        reading_offset = self._get_float_config(
            config,
            "reading_offset",
            "endNum_offset",
            default=0.0
        )
        default_start_angle = self._get_float_config(
            config,
            "default_start_angle",
            default=45.0
        )
        default_range_angle = self._get_float_config(
            config,
            "default_range_angle",
            default=270.0
        )
        validate_mask_line = self._get_bool_config(
            config,
            "validate_mask_line",
            "check_mask_line",
            default=True
        )
        mask_center_threshold_ratio = self._get_float_config(
            config,
            "mask_center_threshold_ratio",
            default=0.10
        )
        auto_zero = self._get_bool_config(
            config,
            "auto_zero",
            "auto_zero_reading",
            default=True
        )
        auto_zero_threshold = self._get_float_config(
            config,
            "auto_zero_threshold",
            "zero_threshold",
            default=0.025
        )
        reading_backend = str(
            config.get(
                "reading_backend",
                config.get("reading_method", config.get("meter_reading_backend", "transformer"))
            ) or "transformer"
        ).strip().lower()
        return_reading_details = self._get_bool_config(
            config,
            "return_reading_details",
            "return_backend_details",
            default=False
        )
        geometry_fallback_to_transformer = self._get_bool_config(
            config,
            "geometry_fallback_to_transformer",
            "fallback_geometry_to_transformer",
            default=False
        )
        residual_calibrator_path = config.get(
            "residual_calibrator_path",
            config.get("geometry_calibrator_path", config.get("calibration_model_path"))
        )
        residual_hybrid_model_path = config.get(
            "residual_hybrid_model_path",
            config.get("geometry_hybrid_model_path", config.get("hybrid_model_path"))
        )
        residual_hybrid_max_abs_delta = self._get_float_config(
            config,
            "residual_hybrid_max_abs_delta",
            "geometry_hybrid_max_abs_delta",
            "hybrid_max_abs_delta",
            default=0.05,
        )
        endNum, resultNum, segPointer, corpImg, origin_crop = self.ZeroShotM.Inference(image,
                                                                     config.get("scaleStart", 0),
                                                                     config.get("scaleEnd", 1.6),
                                                                     confidence=confidence,
                                                                     use_origin_when_no_meter=use_origin_when_no_meter,
                                                                     start_end_distance_threshold=start_end_distance_threshold,
                                                                     start_end_position=start_end_position,
                                                                     snap_out_of_range_pointer=snap_out_of_range_pointer,
                                                                     correction_mode=correction_mode,
                                                                     stretch_x_ratio=stretch_x_ratio,
                                                                     stretch_y_ratio=stretch_y_ratio,
                                                                     reading_offset=reading_offset,
                                                                     default_start_angle=default_start_angle,
                                                                     default_range_angle=default_range_angle,
                                                                     validate_mask_line=validate_mask_line,
                                                                     mask_center_threshold_ratio=mask_center_threshold_ratio,
                                                                     reading_backend=reading_backend,
                                                                     geometry_fallback_to_transformer=geometry_fallback_to_transformer,
                                                                     residual_calibrator_path=residual_calibrator_path,
                                                                     residual_hybrid_model_path=residual_hybrid_model_path,
                                                                     residual_hybrid_max_abs_delta=residual_hybrid_max_abs_delta)

        # endNum 是 0–100 的归一化圆周位置索引，并非角度制数值。
        # resultNum 是量程换算后的读数；segPointer/corpImg 分别是指针掩码和表盘图像。
        if corpImg is None:
            message = getattr(self.ZeroShotM, "last_error_message", None) or "未检测到表盘"
            logger.info(message)
            return {
                "status": False,
                "message": message,
                "result": None,
                "result_pointer_image": None,
                "result_mask_image": None,
            }

        if auto_zero and resultNum < auto_zero_threshold:
            logger.info(f"resultNum:{resultNum:.4f} 小于自动归零阈值 {auto_zero_threshold:.4f}，已置为0.0")
            resultNum = 0.0

        logger.info(f"endNum:{endNum}, resultNum:{resultNum:.2f}")
        show_pointer_image = corpImg
        show_mask_image = segPointer
        pointer_image_base64 = None
        mask_image_base64 = None
        if config.get("result_pointer_image", False) and show_pointer_image is not None:
            _, buffer = cv2.imencode(".jpg", show_pointer_image)
            pointer_image_base64 = base64.b64encode(buffer).decode("utf-8")
        if config.get("result_mask_image", False) and show_mask_image is not None:
            _, buffer = cv2.imencode(".jpg", show_mask_image)
            mask_image_base64 = base64.b64encode(buffer).decode("utf-8")

        selected_backend = reading_backend
        reading_details = getattr(self.ZeroShotM, "_last_reading_details", None) or {}
        if reading_details.get("selected_backend"):
            selected_backend = reading_details.get("selected_backend")

        message = f"归一化指针位置为{endNum}，表盘读数是{resultNum:.2f}"
        result = {
            "status": True,
            "message": message,
            "result": resultNum,
            "result_pointer_image": pointer_image_base64,
            "result_mask_image": mask_image_base64,
        }
        if reading_backend != "transformer" or return_reading_details:
            result["reading_backend"] = selected_backend
        if return_reading_details:
            if reading_details:
                details_payload = {
                    "requested_backend": reading_details.get("requested_backend"),
                    "selected_backend": reading_details.get("selected_backend"),
                    "branch": reading_details.get("branch"),
                    "transformer": reading_details.get("transformer"),
                    "geometry": reading_details.get("geometry"),
                    "geometry_direct": reading_details.get("geometry_direct"),
                    "geometry_direct_v2": reading_details.get("geometry_direct_v2"),
                    "geometry_fusion": reading_details.get("geometry_fusion"),
                    "geometry_fusion_weighted": reading_details.get("geometry_fusion_weighted"),
                    "geometry_fusion_calibrated": reading_details.get("geometry_fusion_calibrated"),
                    "geometry_fusion_weighted_calibrated": reading_details.get("geometry_fusion_weighted_calibrated"),
                    "geometry_hybrid": reading_details.get("geometry_hybrid"),
                    "geometry_legacy": reading_details.get("geometry_legacy"),
                    "selected": reading_details.get("selected"),
                }
                transformer_result = reading_details.get("transformer", {}).get("resultNum")
                geometry_result = reading_details.get("geometry", {}).get("resultNum")
                if transformer_result is not None and geometry_result is not None:
                    details_payload["delta"] = geometry_result - transformer_result
                result["reading_details"] = details_payload
            else:
                result["reading_details"] = None

        return result

    def infer_withCapturing(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        """兼容基础接口；当前生产管线在 ``meterZeroShot`` 内完成预处理。"""
        imageData = kwargs.get('imageData', None)
        logger.info("preprocess finished")
        return imageData

    def postprocess(self):
        """兼容基础接口；响应组装已在 ``infer_image_data_one_pic`` 中完成。"""
        pass




