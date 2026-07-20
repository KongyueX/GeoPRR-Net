import argparse
import os
import sys
from pathlib import Path

import cv2
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.angleDetect.zeroShotMeter import meterZeroShot

'''
问题图片列表:
KZT_HD_20260519170903801.jpg
KZT_HD_20260519170936349.jpg
KZT_HD_20260519171442872.jpg
KZT_HD_20260519171550217.jpg
KZT_HD_20260519171828176.jpg
KZT_HD_20260519171931011.jpg

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the pointer-meter pipeline directly on one image."
    )
    parser.add_argument("image", help="input image path")
    parser.add_argument("--scale-start", type=float, default=0.0)
    parser.add_argument("--scale-end", type=float, default=1.6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    image_path = os.path.abspath(args.image)
    image = cv2.imread(image_path)
    if image is None:
        parser.error(f"cannot decode image: {image_path}")

    filePath = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(os.path.dirname(filePath), "utils")
    u2netWeights = os.path.join(weights_path,
                                "angleDetect\\pointerSeg\\resultSeg\\best.pt")
    yoloWeights = os.path.join(weights_path,
                               "angleDetect\\yoloDetection\\result\\yolo_findMeter.pt")
    meterclipWeights = os.path.join(weights_path,
                                    "angleDetect\\vitTranforms\\result\\best.pt")
    pointWeights = os.path.join(weights_path,
                                "angleDetect\\yoloDetection\\result\\yolo_pointbest.pt")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    ZeroShotM = meterZeroShot(u2netWeights, yoloWeights, meterclipWeights,
                                   pointWeights, device)

    endNum, resultNum, segPointer, corpImg, origin_crop = ZeroShotM.Inference(
        image,
        args.scale_start,
        args.scale_end,
    )

    '''endNum：代表归一化的旋转角度
         resultNum：代表最终表的读数
         segPointer：指针掩码
         corpImg：表盘图像'''
    if corpImg is None:
        logger.info("未检测到表盘")

    logger.info(f"endNum:{endNum}, resultNum:{resultNum:.2f}")
    show_pointer_image = corpImg
    show_mask_image = segPointer
    cv2.imshow("show_pointer_image", show_pointer_image)
    cv2.imshow("show_mask_image", show_mask_image)
    cv2.imshow("origin_crop", origin_crop)
    cv2.waitKey(0)
    # 构建返回消息
    # message = f"旋转角度为{endNum}，表盘读数是{resultNum:.2f}"
