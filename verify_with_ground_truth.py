# -*- coding: utf-8 -*-
"""
用真值验证椭圆校正策略准确性

用户标注的6张图真值，计算每种策略的绝对误差 |策略读数 - 真值|

运行: D:/minconda/envs/pytorch5060/python.exe verify_with_ground_truth.py
"""
import os, sys
import cv2
import numpy as np
import math
import torch
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ)
from utils.angleDetect.zeroShotMeter import meterZeroShot

W = os.path.join(PROJ, "utils", "angleDetect")
WEIGHTS = dict(
    u2netWeights=os.path.join(W, "pointerSeg", "resultSeg", "best_val_iou626.pt"),
    yoloWeights=os.path.join(W, "yoloDetection", "result", "yolo_findMeter.pt"),
    meterclipWeights=os.path.join(W, "vitTranforms", "result", "best.pt"),
    pointWeights=os.path.join(W, "yoloDetection", "result", "yolo_pointbest.pt"),
)
SCALE_START, SCALE_END = 0, 1.6


# 用户标注的真值（真值区间取中值）
GROUND_TRUTH = {
    "13.png": 0.17,
    "234.png": 0.17,
    "253.png": None,  # 过曝，跳过
    "345.png": 0.175,  # 0.17-0.18取中值
    "23423.png": 0.185,  # 0.18-0.19取中值
    "32333.png": 0.46,
}


# ==================== 3种mask生成策略 ====================

def mask_hsv_baseline(crop_bgr):
    """当前HSV方案（baseline）"""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    _, S_, V_ = cv2.split(hsv)
    v_thr = np.percentile(V_, 55)
    mask = ((V_ >= v_thr) & (S_ <= 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    return mask, "HSV Baseline"


def mask_hsv_edge_fusion(crop_bgr):
    """HSV + Canny边缘融合"""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    _, S_, V_ = cv2.split(hsv)
    v_thr = np.percentile(V_, 55)
    mask_hsv = ((V_ >= v_thr) & (S_ <= 90)).astype(np.uint8) * 255
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blur, 50, 150)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask_hsv)
    edges_dilated = cv2.dilate(edges_masked, np.ones((3, 3), np.uint8), iterations=1)
    mask_fused = cv2.bitwise_or(mask_hsv, edges_dilated)
    mask_fused = cv2.morphologyEx(mask_fused, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    return mask_fused, "HSV+Edge Fusion"


def mask_edge_direct(crop_bgr):
    """Canny边缘直接拟合"""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges_closed)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [c], -1, 255, -1)
    return mask, "Edge Direct"


def fit_ellipse_from_mask(mask, crop_bgr):
    """从mask提取椭圆，复用当前的质量校验逻辑"""
    h, w = crop_bgr.shape[:2]
    area_img = float(h * w)
    cx0, cy0 = w / 2.0, h / 2.0

    num, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    if num < 2:
        return None, None, "No components"

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
        return None, None, "No valid component"

    comp = (lab == best).astype(np.uint8) * 255
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, "No contours"

    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20:
        return None, None, "Too few points"

    ellipse = cv2.fitEllipse(c)
    (ecx, ecy), (d1, d2), _ = ellipse
    a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0
    if b < 1:
        return None, None, "Invalid axes"

    ratio = b / a
    fill = math.pi * a * b / area_img
    centric = math.hypot(ecx - cx0, ecy - cy0) / math.hypot(cx0, cy0)

    if ratio < 0.45 or fill < 0.12 or fill > 1.0 or centric > 0.35:
        reason = f"Quality fail (r={ratio:.2f},f={fill:.2f},c={centric:.2f})"
        return None, None, reason

    return ellipse, comp, "OK"


def make_patched_fit(strategy_fn, fit_ellipse_fn):
    """生成一个用指定策略的_fit_face_ellipse替代函数"""
    def patched(crop_bgr):
        mask, _ = strategy_fn(crop_bgr)
        ellipse, comp, status = fit_ellipse_fn(mask, crop_bgr)
        if ellipse is None:
            return None, mask
        return ellipse, comp
    return patched


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device= {device}\n")

    model = meterZeroShot(device=device, **WEIGHTS)

    data_dir = os.path.join(PROJ, "data")
    strategies = [
        ("off", None),  # 无矫正基线
        ("HSV", mask_hsv_baseline),
        ("Fusion", mask_hsv_edge_fusion),
        ("Edge", mask_edge_direct),
    ]

    orig_fit = model._fit_face_ellipse

    results_table = []

    for name, truth in GROUND_TRUTH.items():
        if truth is None:
            print(f"[跳过] {name} - 过曝无真值")
            continue

        img_path = os.path.join(data_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[错误] {name} - 读取失败")
            continue

        row = {"name": name, "truth": truth}

        for label, strategy in strategies:
            if label == "off":
                # off基线
                _, res, _, _, _ = model.Inference(img, SCALE_START, SCALE_END,
                                                   correction_mode="off")
            else:
                # 用指定策略矫正
                model._fit_face_ellipse = make_patched_fit(strategy, fit_ellipse_from_mask)
                try:
                    _, res, _, _, _ = model.Inference(img, SCALE_START, SCALE_END,
                                                     correction_mode="ellipse")
                except Exception:
                    res = None
                model._fit_face_ellipse = orig_fit

            if res is not None:
                error = abs(float(res) - truth)
                row[label] = (res, error)
            else:
                row[label] = (None, None)

        results_table.append(row)

        # 打印这行结果
        print(f"{name:<20} 真值={truth:.3f}")
        for label, _ in strategies:
            res, err = row.get(label, (None, None))
            if res is not None and err is not None:
                print(f"  {label:<10} 读数={res:.4f}  误差={err:.4f}")
            else:
                print(f"  {label:<10} 读数=—       误差=—")
        print()

    # 汇总统计
    print("=" * 70)
    print("汇总统计（平均绝对误差，越小越准）")
    print("=" * 70)

    summary = {}
    for label, _ in strategies:
        errors = [row[label][1] for row in results_table if label in row and row[label][1] is not None]
        if errors:
            mean_err = sum(errors) / len(errors)
            max_err = max(errors)
            summary[label] = (len(errors), mean_err, max_err)
        else:
            summary[label] = (0, None, None)

    for label, (n, mean_err, max_err) in summary.items():
        if mean_err is not None:
            print(f"{label:<15} N={n}  平均误差={mean_err:.4f}  最大误差={max_err:.4f}")
        else:
            print(f"{label:<15} N={n}  无有效数据")

    print("\n结论:")
    valid = {k: v for k, v in summary.items() if v[1] is not None}
    if valid:
        best = min(valid.items(), key=lambda x: x[1][1])
        print(f"平均误差最低: {best[0]} (误差={best[1][1]:.4f})")


if __name__ == "__main__":
    main()
