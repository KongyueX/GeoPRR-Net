# services/DataProcessService.py
import base64
import time
from pathlib import Path

import cv2
from loguru import logger
import numpy as np


class DataProcessService:
    def __init__(self):
        self.isCapturingDict = {}
        self._cameraProcessService = None

    @property
    def cameraProcessService(self):
        # 相机服务含多进程资源，首次收到相机请求时再初始化。
        if self._cameraProcessService is None:
            from services.CameraProcessService import CameraProcessService
            self._cameraProcessService = CameraProcessService()
        return self._cameraProcessService

    def get_image(self, *args, **kwargs):
        imageType = kwargs.get('imageType', None)
        inferData = kwargs.get('inferData', None)
        cameraTimeout = kwargs.get('cameraTimeout', 10)
        config:dict = kwargs.get('config', {})
        if inferData is None:
            return False, "Error: inferData is None"
        if imageType is None:
            return False, "Error: imageType is None"
        elif imageType == 'filepath':
            return self.get_image_from_file_path(inferData)
        elif imageType == 'base64':
            return self.get_image_from_base64(inferData)
        elif imageType == 'cameraId':
            if self.isCapturingDict.get(str(inferData),False):
                return self.cameraProcessService.get_image(camera_id=inferData,timeout=cameraTimeout)
            else:
                ret, message = self.cameraProcessService.start_camera_capture(camera_id=inferData,
                                                                             camera_type=config.get("cameraType", None))
                if not ret:
                    return False, message
                self.isCapturingDict[str(inferData)] = True
                logger.info(message)
                time.sleep(5)
                return self.cameraProcessService.get_image(camera_id=inferData,timeout=cameraTimeout)
        else:
            return False, "Error: imageType is not supported, please choose in [filepath, base64, cameraId]"

    def get_image_from_file_path(self, file_path):
        """
        从文件路径获取图像
        Args:
            file_path (str): 文件路径
        Returns:
            tuple: (success: bool, result: Union[np.ndarray, str])
                   success为True时返回图像数组,False时返回错误信息
        """
        try:
            normalized_path = Path(str(file_path)).expanduser()
            if not normalized_path.is_absolute():
                normalized_path = (Path.cwd() / normalized_path).resolve()
            else:
                normalized_path = normalized_path.resolve()
        except Exception as e:
            return False, f"Error: invalid file_path - {str(e)}"

        if not normalized_path.exists():
            return False, f"Error: file_path is not exists - {normalized_path}"

        image = cv2.imread(str(normalized_path))
        if image is None:
            return False, f"Error: failed to read image - {normalized_path}"

        return True, image

    def get_image_from_base64(self, base64_str):
        """
        从base64字符串获取图像
        Args:
            base64_str (str): base64编码的图像字符串
        Returns:
            tuple: (success: bool, result: Union[np.ndarray, str])
                   success为True时返回图像数组,False时返回错误信息
        """
        try:
            if not base64_str or not isinstance(base64_str, str):
                return False, "Error: Invalid base64 string input"

            byte_data = base64.b64decode(base64_str)
            if len(byte_data) == 0:
                return False, "Error: Empty image data"

            encode_image = np.frombuffer(byte_data, dtype=np.uint8)
            # OpenCV 在整个服务中统一使用 BGR，不在输入层转换为 RGB。
            img_array = cv2.imdecode(encode_image, cv2.IMREAD_COLOR)

            if img_array is None:
                return False, "Error: Failed to decode image"

            return True, img_array

        except base64.binascii.Error:
            return False, "Error: Invalid base64 encoding"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def endCapture(self, *args, **kwargs):
        cameraTimeout = kwargs.get('cameraTimeout', 10)
        cameraId = kwargs.get('cameraId', None)
        if cameraId is None:
            return False, "Error: cameraId is None"

        # 更新本地状态
        cameraIdStr = str(cameraId)
        if cameraIdStr in self.isCapturingDict:
            self.isCapturingDict[cameraIdStr] = False

        return self.cameraProcessService.stop_camera_capture(cameraId, timeout=cameraTimeout)
