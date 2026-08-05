#-O 日后必定搭建属于自己的yolo模型
import os

import cv2
import numpy as np
from ultralytics import YOLO
#import logging
import math

filePath = os.path.dirname(os.path.abspath(__file__))

from .pointGet import demonstrate_intersections

class targetDetectModel:
    def __init__(self, model_path):
        """初始化目标检测模型
        Args:
            model_path (str): 模型文件相对路径
        Raises:
            FileNotFoundError: 当模型文件不存在时
            RuntimeError: 当模型加载失败时
        """
        self.model_path = os.path.join(filePath, model_path)
        self.fire_detection_model = None
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"路径缺失: {self.model_path}")
                
            self.load_target_detection_model()
            #logging.info(f"成功加载目标检测模型: {self.model_path}")
            
        except Exception as e:
            #logging.error(f"模型加载失败: {self.model_path}，{str(e)}")
            raise

    def load_target_detection_model(self):
        """加载YOLO模型"""
        try:
            self.fire_detection_model = YOLO(self.model_path)
            # 验证模型是否加载成功
            if self.fire_detection_model is None:
                raise RuntimeError("该模型权重不属于YOLO目标检测模型")
                
        except Exception as e:
            #logging.error(f"模型加载失败: {str(e)}")
            raise

    def _predict(self, image, confidence=None):
        if self.fire_detection_model is None:
            self.load_target_detection_model()

        predict_kwargs = {"verbose": False}
        if confidence is not None:
            predict_kwargs["conf"] = float(confidence)

        return self.fire_detection_model.predict(image, **predict_kwargs)

    def target_detection(self, image, confidence=None):
        """执行目标检测
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            tuple: (confidences, dt_boxes, img_crop_list, class_ids)
            
        Raises:
            ValueError: 当输入图像无效时
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid input image")
            
        try:
            # 使用模型进行预测
            result = self._predict(image, confidence)
            confidences = []
            dt_boxes = []
            img_crop_list = []
            class_ids = []

            # 检查是否有检测结果
            if len(result) == 0:
                #logging.warning("No detection results")
                return [], [], [], []
                
            # 获取第一个（也是唯一一个）检测结果对象
            detection_result = result[0]
            
            # 检查是否有检测框
            if detection_result.boxes is None or len(detection_result.boxes) == 0:
                #logging.warning("No targets detected")
                return [], [], [], []
            
            # print(f"检测到 {len(detection_result.boxes)} 个目标")
            
            # 遍历所有检测框
            for i in range(len(detection_result.boxes)):
                # 获取置信度
                confidence = detection_result.boxes.conf[i].item()
                
                # 获取类别ID
                cls_id = int(detection_result.boxes.cls[i].item())
                
                # 获取边界框坐标
                box = detection_result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box[:4])

                # 边界检查
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x1 >= x2 or y1 >= y2:
                    #logging.warning(f"Invalid box coordinates: ({x1},{y1},{x2},{y2})")
                    continue

                # 裁剪图像
                img_crop = image[y1:y2, x1:x2]
                
                # 创建坐标数组（用于后续处理）
                coordinates = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.float32)

                # 添加到结果列表
                confidences.append(confidence)
                dt_boxes.append(coordinates)
                img_crop_list.append(img_crop)
                class_ids.append(cls_id)
                
                # 打印调试信息
                # print(f"目标 {i+1}: 类别={cls_id}, 置信度={confidence:.3f}, 坐标=({x1},{y1},{x2},{y2})")

            return confidences, dt_boxes, img_crop_list, class_ids

        except Exception as e:
            #logging.error(f"目标检测出错: {e}")
            return [], [], [], []

    def center_find(self, image, classId=None, confidence=None):
        """计算目标中心坐标
        de
        Returns:
            tuple: (x, y, confidences) 或 (None, None, []) 当无目标时
        """

        try:
            confidences, dt_boxes, img_crop_list, class_ids = self.target_detection(image, confidence)
            if not dt_boxes:
                return None, None

            if classId is None:
            #-O 这里默认返回的应该是置信度最大的点    
                box = dt_boxes[0]
        
            else:
                index = class_ids.index(classId)
                box = dt_boxes[index]

            x = 0.5 * (box[2][0] - box[0][0]) + box[0][0]
            y = 0.5 * (box[2][1] - box[0][1]) + box[0][1]

            center = (x, y)

            return center, confidences
            
        except Exception as e:
            #logging.error(f"没有寻找到中点: {str(e)}")
            return None, None
        
    def letterbox(self,img, new_shape=640, color=(255, 255, 255)):
        """
        YOLO风格的Letterbox缩放（默认白色填充）
        Args:
            img: 输入图像 (HWC格式的numpy数组)
            new_shape: 目标尺寸 (int或元组, 如640或(640,640))
            color: 填充颜色 (BGR格式, 默认白色)
        Returns:
            img: 缩放并填充后的图像
        """
        # 统一目标尺寸格式
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 原始图像尺寸
        h, w = img.shape[:2]
        
        # 计算缩放比例并等比例缩放
        scale = min(new_shape[0] / h, new_shape[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 计算填充位置（居中）
        top = (new_shape[0] - new_h) // 2
        bottom = new_shape[0] - new_h - top
        left = (new_shape[1] - new_w) // 2
        right = new_shape[1] - new_w - left
        
        # 白色填充
        letterbox_img = cv2.copyMakeBorder(
            resized_img, 
            top, bottom, left, right, 
            cv2.BORDER_CONSTANT, 
            value=color
        )
        return letterbox_img


    def image_crop(self, image, size=0, confidence=None, use_origin_when_no_meter=False):
        """裁剪目标区域
        
        Returns:
            np.ndarray: 裁剪后的图像或原图(当检测失败时)
        """
        try:
            result = self._predict(image, confidence)
            # 用于存储所有检测结果
            all_confidences = []
            all_boxes = []
            all_crops = []
            all_class_ids = []
            
            for result_single in result[0]:
                if len(result_single.boxes) == 0:  # 无检测结果
                    continue
                    
                # 遍历该检测结果的所有框（而不是只取第一个）
                for box_idx in range(len(result_single.boxes)):
                    # 获取类别ID和置信度
                    cls_id = int(result_single.boxes.cls[box_idx].item())
                    confidence = result_single.boxes.conf[box_idx].item()  # 置信度（float）
                    
                    # 获取边界框坐标
                    box = result_single.boxes.xyxy[box_idx].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # 边界检查
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1-3*size), max(0, y1-size)
                    x2, y2 = min(w, x2+3*size), min(h, y2+size)
                    
                    if x1 >= x2 or y1 >= y2:
                        #logging.warning(f"Invalid box coordinates: ({x1},{y1},{x2},{y2})")
                        continue
                    
                    # 裁剪图像
                    img_crop = image[y1:y2, x1:x2]
                    
                    # 存储检测结果
                    all_confidences.append(confidence)
                    all_boxes.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
                    all_crops.append(img_crop)
                    all_class_ids.append(cls_id)
                    
            # 如果没有检测到目标
            if not all_confidences:
                #logging.warning("No targets detected")
                if use_origin_when_no_meter:
                    return [], [], [image], [], 0
                return [], [], [], [], None
            
            # 找到置信度最高的索引
            best_idx = np.argmax(all_confidences)
            
            # 返回置信度最高的检测结果（保持原格式）
            return (
                    all_confidences,  # 直接返回str
                    all_boxes,                 # 直接返回np.array
                    all_crops,                 # 直接返回np.array（图像）
                    all_class_ids,         # 直接返回str
                    best_idx
            )
                        
        except Exception as e:
            #logging.error(f"Detection failed: {str(e)}")
            if use_origin_when_no_meter:
                return [], [], [image], [], 0
            return [], [], [], [], None

    def _find_gauge_reference_points(self, corpImg, confidence=None):
        """
        检测表盘校正所需的三个语义点。

        classId=0: 表盘中心点
        classId=2: 左下角 0 点
        classId=1: 右下角量程终点
        """
        if len(corpImg.shape) == 3:
            corpImgGray = cv2.cvtColor(corpImg, cv2.COLOR_BGR2GRAY)
        else:
            corpImgGray = corpImg.copy()
        corpImgGray = np.stack((corpImgGray,) * 3, axis=-1)

        confidences, dt_boxes, _, class_ids = self.target_detection(corpImgGray, confidence)
        best_points = {}

        for point_confidence, box, class_id in zip(confidences, dt_boxes, class_ids):
            if class_id not in (0, 1, 2):
                continue
            if class_id in best_points and point_confidence <= best_points[class_id][0]:
                continue

            x = 0.5 * (box[2][0] - box[0][0]) + box[0][0]
            y = 0.5 * (box[2][1] - box[0][1]) + box[0][1]
            best_points[class_id] = (point_confidence, np.float32([x, y]))

        if not all(class_id in best_points for class_id in (0, 1, 2)):
            return None

        point_center = best_points[0][1]
        point_end = best_points[1][1]
        point_start = best_points[2][1]

        return point_center, point_start, point_end

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

    @staticmethod
    def _cross_value(center, point_a, point_b):
        vec_a = point_a - center
        vec_b = point_b - center
        return float(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])

    def _normalize_reference_points_for_layout(
            self,
            point_center,
            point_start,
            point_end,
            start_end_position
    ):
        start_left = self._is_start_expected_on_left(start_end_position)

        start_on_expected_side = point_start[0] <= point_end[0] if start_left else point_start[0] >= point_end[0]
        if not start_on_expected_side:
            point_start, point_end = point_end, point_start

        return point_center, point_start, point_end

    def _semantic_points_to_upright_meter(self, corpImg, point_center, point_start, point_end, start_end_position):
        """
        使用中心点、左下 0 点、右下量程终点做仿射校正。

        默认标准输出中 0 点落在左下 45 度方向，量程终点落在右下 45 度方向。
        当配置为 start_right_end_left 时，两者目标位置相反。
        """
        height, width = corpImg.shape[:2]
        size = int(max(height, width))
        if size <= 0:
            return corpImg

        point_center, point_start, point_end = self._normalize_reference_points_for_layout(
            point_center,
            point_start,
            point_end,
            start_end_position,
        )

        center_to_start = point_start - point_center
        center_to_end = point_end - point_center
        start_distance = float(np.linalg.norm(center_to_start))
        end_distance = float(np.linalg.norm(center_to_end))
        min_valid_distance = max(8.0, min(height, width) * 0.08)

        if start_distance < min_valid_distance or end_distance < min_valid_distance:
            return corpImg

        cross = abs(float(
            center_to_start[0] * center_to_end[1] -
            center_to_start[1] * center_to_end[0]
        ))
        if cross < min_valid_distance * min_valid_distance:
            return corpImg

        output_center = np.float32([size / 2.0, size / 2.0])
        # 0点/终点通常在刻度附近，不一定是圆盘最外边界；留足外圈边距避免裁切。
        output_radius = size * 0.35
        diagonal_offset = output_radius / math.sqrt(2.0)

        if self._is_start_expected_on_left(start_end_position):
            target_start = output_center + np.float32([-diagonal_offset, diagonal_offset])
            target_end = output_center + np.float32([diagonal_offset, diagonal_offset])
        else:
            target_start = output_center + np.float32([diagonal_offset, diagonal_offset])
            target_end = output_center + np.float32([-diagonal_offset, diagonal_offset])

        source_cross = self._cross_value(point_center, point_start, point_end)
        target_cross = self._cross_value(output_center, target_start, target_end)
        if source_cross * target_cross < 0:
            point_start, point_end = point_end, point_start

        src_points = np.float32([point_center, point_start, point_end])
        dst_points = np.float32([output_center, target_start, target_end])

        matrix = cv2.getAffineTransform(src_points, dst_points)
        border_value = (255, 255, 255) if len(corpImg.shape) == 3 else 255
        return cv2.warpAffine(
            corpImg,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    def ransacFun(self, corpImg, confidence=None, start_end_position="start_left_end_right"):
        """
        圆表倾斜校正入口。

        YOLO 检出的三个点分别是：
        - 表盘中心点 classId=0
        - 左下角 0 点 classId=2
        - 右下角量程终点 classId=1

        这里用三点仿射变换把表盘归一化到标准姿态，避免纯椭圆拟合根据
        轮廓长轴方向产生过大的旋转角度。
        """
        reference_points = self._find_gauge_reference_points(corpImg, confidence=confidence)
        if reference_points is None:
            return corpImg

        point_center, point_start, point_end = reference_points
        return self._semantic_points_to_upright_meter(
            corpImg,
            point_center,
            point_start,
            point_end,
            start_end_position,
        )

    def ransacFun_backup(self, corpImg, confidence=None):
        height, width = corpImg.shape[0:2]

        #-O 这里返还的点是基于裁剪的图像的
        #-O 这里的推理速度也比较慢，需要进行推理加速
        corpImgGray = cv2.cvtColor(corpImg, cv2.COLOR_BGR2GRAY)
        corpImgGray = np.stack((corpImgGray,) * 3, axis=-1)
        center, confidences  = self.center_find(corpImgGray, classId=1, confidence=confidence)
        center_b, confidences  = self.center_find(corpImgGray, classId=2, confidence=confidence)
        pointCenter, confidences = self.center_find(corpImgGray, classId=0, confidence=confidence)
        # print(pointCenter)

        center = tuple(map(int, center))
        center_b = tuple(map(int, center_b))
        pointCenter = tuple(map(int, pointCenter))
        show_image = corpImgGray.copy()
        cv2.circle(show_image, (pointCenter[0], pointCenter[1]), 5, (0, 0, 255), -1)
        cv2.circle(show_image, (center[0], center[1]), 5, (255, 0, 0), -1)
        cv2.circle(show_image, (center_b[0], center_b[1]), 5, (255, 0, 0), -1)

        changeX = demonstrate_intersections(center, center_b, pointCenter, corpImg)
        src_points = np.float32([changeX[key] for key in ['up', 'right', 'down', 'left']])

        #-O 输出图像尺寸
        size = width  
        # 图像一半的宽度
        center = r = size / 2  

        dst_points = np.float32([
                                [center, center - r],  # 12 点钟位置
                                [center + r, center],  # 3 点钟位置
                                [center, center + r],  # 6 点钟位置
                                [center - r, center]   # 9 点钟位置
        ])


        #-O 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        #-O 应用透视变换的到变化后的图
        corrected = cv2.warpPerspective(corpImg, M, (size, size))

        return corrected

def calculate_angle(C, P, angle=180):
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



if __name__ == "__main__":
    weights = "yoloDetection\\result\\yolo_findMeter.pt"
    weights_2 = "yoloDetection\\result\\yolo_pointbest.pt"

    meterDetect = targetDetectModel(weights)
    pointerDetect = targetDetectModel(weights_2)

    img = cv2.imread("testImg\\49957dd8122427ee2b08bf9eb1391c20.jpg")

    all_confidences,  all_boxes, all_crops, ll_class_ids, best_idx = meterDetect.image_crop(img)

    corpImg = all_crops[best_idx]

    corrected = pointerDetect.ransacFun(corpImg)

    cv2.namedWindow("demo1", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("demo1",corrected)
    cv2.waitKey(0)

