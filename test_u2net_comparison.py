"""
对比测试新旧 U2Net 模型效果
展示中间过程：表盘检测 -> 图像矫正 -> 指针分割 -> 最终读数
"""
import os
import cv2
import numpy as np
from loguru import logger
from utils.angleDetect.zeroShotMeter import meterZeroShot

def create_comparison_display(old_result, new_result, image_name):
    """
    创建对比显示图
    """
    # 提取结果
    old_endNum, old_resultNum, old_segPointer, old_corpImg, old_origin_crop = old_result
    new_endNum, new_resultNum, new_segPointer, new_corpImg, new_origin_crop = new_result

    # 创建对比图
    rows = []

    # 标题行
    title_height = 60
    title_img = np.ones((title_height, 1200, 3), dtype=np.uint8) * 255
    cv2.putText(title_img, f"Image: {image_name}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    rows.append(title_img)

    # 原始裁剪图（两个应该一样）
    if old_origin_crop is not None:
        origin_display = old_origin_crop.copy()
        h, w = origin_display.shape[:2]
        if w > 600:
            scale = 600 / w
            origin_display = cv2.resize(origin_display, None, fx=scale, fy=scale)
        label_img = np.ones((40, origin_display.shape[1], 3), dtype=np.uint8) * 240
        cv2.putText(label_img, "Original Meter Crop", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        origin_section = np.vstack([label_img, origin_display])
        rows.append(origin_section)

    # 对比行：旧模型 vs 新模型
    comparison_parts = []

    # 旧模型结果
    if old_corpImg is not None and old_segPointer is not None:
        # 确保图像尺寸一致
        old_corp_display = old_corpImg.copy()
        old_seg_display = old_segPointer.copy()

        if len(old_seg_display.shape) == 2:
            old_seg_display = cv2.cvtColor(old_seg_display, cv2.COLOR_GRAY2BGR)

        # 调整尺寸
        target_h = 300
        h, w = old_corp_display.shape[:2]
        scale = target_h / h
        old_corp_display = cv2.resize(old_corp_display, None, fx=scale, fy=scale)
        old_seg_display = cv2.resize(old_seg_display, None, fx=scale, fy=scale)

        # 拼接表盘和掩码
        old_combined = np.hstack([old_corp_display, old_seg_display])

        # 添加标签和读数
        label_h = 80
        label_img = np.ones((label_h, old_combined.shape[1], 3), dtype=np.uint8) * 200
        cv2.putText(label_img, "OLD MODEL (best.pt)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(label_img, f"Reading: {old_resultNum:.3f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        old_section = np.vstack([label_img, old_combined])
        comparison_parts.append(old_section)

    # 新模型结果
    if new_corpImg is not None and new_segPointer is not None:
        new_corp_display = new_corpImg.copy()
        new_seg_display = new_segPointer.copy()

        if len(new_seg_display.shape) == 2:
            new_seg_display = cv2.cvtColor(new_seg_display, cv2.COLOR_GRAY2BGR)

        # 调整尺寸
        target_h = 300
        h, w = new_corp_display.shape[:2]
        scale = target_h / h
        new_corp_display = cv2.resize(new_corp_display, None, fx=scale, fy=scale)
        new_seg_display = cv2.resize(new_seg_display, None, fx=scale, fy=scale)

        # 拼接表盘和掩码
        new_combined = np.hstack([new_corp_display, new_seg_display])

        # 添加标签和读数
        label_h = 80
        label_img = np.ones((label_h, new_combined.shape[1], 3), dtype=np.uint8) * 200
        cv2.putText(label_img, "NEW MODEL (best_val_iou.pt)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(label_img, f"Reading: {new_resultNum:.3f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        new_section = np.vstack([label_img, new_combined])
        comparison_parts.append(new_section)

    # 水平拼接对比
    if len(comparison_parts) == 2:
        # 确保高度一致
        max_h = max(part.shape[0] for part in comparison_parts)
        padded_parts = []
        for part in comparison_parts:
            if part.shape[0] < max_h:
                pad_h = max_h - part.shape[0]
                padding = np.ones((pad_h, part.shape[1], 3), dtype=np.uint8) * 255
                part = np.vstack([part, padding])
            padded_parts.append(part)
        comparison_row = np.hstack(padded_parts)
        rows.append(comparison_row)

    # 差异统计行
    if old_resultNum is not None and new_resultNum is not None:
        diff = new_resultNum - old_resultNum
        stats_img = np.ones((80, 1200, 3), dtype=np.uint8) * 250
        cv2.putText(stats_img, f"Difference: {diff:.4f} ({diff/old_resultNum*100 if old_resultNum != 0 else 0:.2f}%)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0) if abs(diff) > 0.05 else (0, 128, 0), 2)
        cv2.putText(stats_img, f"Old: {old_resultNum:.4f}  |  New: {new_resultNum:.4f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        rows.append(stats_img)

    # 垂直拼接所有行
    if rows:
        # 确保所有行宽度一致
        max_w = max(row.shape[1] for row in rows)
        padded_rows = []
        for row in rows:
            if row.shape[1] < max_w:
                pad_w = max_w - row.shape[1]
                padding = np.ones((row.shape[0], pad_w, 3), dtype=np.uint8) * 255
                row = np.hstack([row, padding])
            padded_rows.append(row)
        final_display = np.vstack(padded_rows)
        return final_display

    return None


def test_single_image(image_path, old_model, new_model, config):
    """
    测试单张图片
    """
    image_name = os.path.basename(image_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"测试图片: {image_name}")
    logger.info(f"{'='*60}")

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return None

    # 旧模型推理
    logger.info(">>> 旧模型 (best.pt) 推理中...")
    old_result = old_model.Inference(
        image,
        scaleStart=config['scaleStart'],
        scaleEnd=config['scaleEnd'],
        confidence=config.get('confidence'),
        correction_mode=config.get('correction_mode', 'ransacFun'),
    )
    old_endNum, old_resultNum, old_segPointer, old_corpImg, old_origin_crop = old_result

    if old_corpImg is not None:
        logger.info(f"旧模型结果 - endNum: {old_endNum}, resultNum: {old_resultNum:.4f}")
    else:
        logger.warning("旧模型未检测到表盘")

    # 新模型推理
    logger.info(">>> 新模型 (best_val_iou.pt) 推理中...")
    new_result = new_model.Inference(
        image,
        scaleStart=config['scaleStart'],
        scaleEnd=config['scaleEnd'],
        confidence=config.get('confidence'),
        correction_mode=config.get('correction_mode', 'ransacFun'),
    )
    new_endNum, new_resultNum, new_segPointer, new_corpImg, new_origin_crop = new_result

    if new_corpImg is not None:
        logger.info(f"新模型结果 - endNum: {new_endNum}, resultNum: {new_resultNum:.4f}")
    else:
        logger.warning("新模型未检测到表盘")

    # 对比分析
    if old_resultNum is not None and new_resultNum is not None:
        diff = new_resultNum - old_resultNum
        logger.info(f">>> 差异: {diff:.4f} ({diff/old_resultNum*100 if old_resultNum != 0 else 0:.2f}%)")

    return old_result, new_result


if __name__ == '__main__':
    # 配置路径
    filePath = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(filePath, "utils")

    # 旧模型权重
    old_u2net_weights = os.path.join(weights_path,
                                      "angleDetect/pointerSeg/resultSeg/best.pt")
    # 新模型权重
    new_u2net_weights = os.path.join(weights_path,
                                      "angleDetect/pointerSeg/resultSeg/best_val_iou.pt")

    # 共享的其他模型权重
    yolo_weights = os.path.join(weights_path,
                                "angleDetect/yoloDetection/result/yolo_findMeter.pt")
    meterclip_weights = os.path.join(weights_path,
                                     "angleDetect/vitTranforms/result/best.pt")
    point_weights = os.path.join(weights_path,
                                 "angleDetect/yoloDetection/result/yolo_pointbest.pt")

    device = "cpu"

    # 加载旧模型
    logger.info("加载旧模型 (best.pt)...")
    old_model = meterZeroShot(old_u2net_weights, yolo_weights, meterclip_weights,
                               point_weights, device)

    # 加载新模型
    logger.info("加载新模型 (best_val_iou.pt)...")
    new_model = meterZeroShot(new_u2net_weights, yolo_weights, meterclip_weights,
                               point_weights, device)

    # 测试配置
    config = {
        'scaleStart': 0,
        'scaleEnd': 1.6,
        'confidence': None,
        'correction_mode': 'ransacFun',
    }

    # 测试图片
    test_images = [
        "data/KZT_HD_20260616150323688.jpg",
        "data/KZT_HD_20260616150729647.jpg",
        "data/KZT_HD_20260616151225999.jpg",
    ]

    # 确保输出目录存在
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 逐个测试
    for idx, img_path in enumerate(test_images):
        if not os.path.exists(img_path):
            logger.warning(f"图片不存在: {img_path}")
            continue

        # 测试图片
        old_result, new_result = test_single_image(img_path, old_model, new_model, config)

        # 创建对比图
        comparison_img = create_comparison_display(old_result, new_result, os.path.basename(img_path))

        if comparison_img is not None:
            # 保存对比图
            output_path = os.path.join(output_dir, f"comparison_{idx+1}_{os.path.basename(img_path)}")
            cv2.imwrite(output_path, comparison_img)
            logger.info(f"对比图已保存: {output_path}")

            # 显示对比图
            cv2.imshow(f"Comparison {idx+1}", comparison_img)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:  # q 或 ESC 退出
                break
            cv2.destroyAllWindows()

    logger.info("\n" + "="*60)
    logger.info("测试完成！")
    logger.info(f"对比图已保存到: {output_dir}/")
    logger.info("="*60)
