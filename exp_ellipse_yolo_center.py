# -*- coding: utf-8 -*-
"""
椭圆矫正v3：YOLO中心点约束版本

改进点：用YOLO检测的表盘中心(classId=0)替换轮廓拟合的椭圆中心，
       保留拟合的长短轴和角度信息，提升中心定位精度。
"""
import os, sys, math
import numpy as np
import cv2
import torch
from loguru import logger

logger.remove(); logger.add(sys.stderr, level="INFO")
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ)
from utils.angleDetect.zeroShotMeter import meterZeroShot

W = os.path.join(PROJ, "utils", "angleDetect")
u2net  = os.path.join(W, "pointerSeg", "resultSeg", "best.pt")
yolo   = os.path.join(W, "yoloDetection", "result", "yolo_findMeter.pt")
mclip  = os.path.join(W, "vitTranforms", "result", "best.pt")
points = os.path.join(W, "yoloDetection", "result", "yolo_pointbest.pt")
SCALE_START, SCALE_END = 0, 1.6
DEBUG_DIR = os.path.join(PROJ, "test_results", "ellipse_yolo_center")


def warp_pitch(img, phi_deg, axis='x', f_ratio=1.0, d_ratio=2.0):
    """模拟俯仰角拍摄"""
    h, w = img.shape[:2]
    f, d = w * f_ratio, w * d_ratio
    phi = math.radians(phi_deg)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    pts = np.float32([[-w/2,-h/2,0],[w/2,-h/2,0],[w/2,h/2,0],[-w/2,h/2,0]])
    c, s = math.cos(phi), math.sin(phi)
    R = np.float32([[1,0,0],[0,c,-s],[0,s,c]]) if axis=='x' else np.float32([[c,0,s],[0,1,0],[-s,0,c]])
    r = (R @ pts.T).T; r[:,2] += d
    dst = np.zeros((4,2),np.float32)
    dst[:,0] = f*r[:,0]/r[:,2] + w/2
    dst[:,1] = f*r[:,1]/r[:,2] + h/2
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w,h), borderValue=(0,0,0))


def run_and_extract_debug(model, img, mode="off"):
    """运行推理并提取调试信息"""
    out = model.Inference(img, SCALE_START, SCALE_END, correction_mode=mode)
    debug = getattr(model, '_correction_debug', {})
    return out[0], out[1], debug


def save_debug_comparison(name, corpImg, debug, output_path):
    """保存调试对比图"""
    panels = [corpImg]
    labels = ["原图"]

    if debug.get("face_mask") is not None:
        panels.append(cv2.cvtColor(debug["face_mask"], cv2.COLOR_GRAY2BGR))
        labels.append("表盘面")

    if debug.get("ellipse_overlay") is not None:
        ov = debug["ellipse_overlay"].copy()
        # 如果有YOLO中心，标记出来
        if debug.get("yolo_center") is not None:
            cx, cy = debug["yolo_center"]
            cv2.circle(ov, (int(cx), int(cy)), 5, (255, 0, 0), -1)  # 蓝色点
            cv2.putText(ov, "YOLO", (int(cx)+10, int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        panels.append(ov)
        labels.append("拟合椭圆")

    if debug.get("rectified") is not None:
        panels.append(debug["rectified"])
        labels.append("矫正后")

    # 统一高度拼接
    h = 200
    resized = []
    for i, p in enumerate(panels):
        scale = h / p.shape[0]
        w_new = int(p.shape[1] * scale)
        r = cv2.resize(p, (w_new, h))
        # 添加标签
        cv2.putText(r, labels[i], (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        resized.append(r)

    result = np.hstack(resized)
    cv2.imwrite(output_path, result)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = meterZeroShot(u2net, yolo, mclip, points, device)

    data_dir = os.path.join(PROJ, "data")
    imgs = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".jpg",".png",".jpeg"))])

    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 测试场景
    scenarios = [
        ("frontal", lambda img: img, "正面拍摄"),
        ("pitch30", lambda img: warp_pitch(img, 30), "俯视30度"),
        ("pitch45", lambda img: warp_pitch(img, 45), "俯视45度"),
    ]

    agg = {"ellipse": []}

    print(f"\n{'='*80}")
    print(f"测试椭圆矫正 + YOLO中心点约束")
    print(f"{'='*80}\n")

    for i, name in enumerate(imgs[:5]):  # 只测试前5张
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None:
            continue

        print(f"[{i+1}] {name}")

        # 基线：正面无矫正
        b_end, b_res, _ = run_and_extract_debug(model, img, "off")
        if b_end is None:
            print(f"  基线失败，跳过")
            continue

        for scenario_name, transform_fn, desc in scenarios:
            test_img = transform_fn(img)

            # 测试ellipse模式（已包含YOLO中心约束）
            t_end, t_res, debug = run_and_extract_debug(model, test_img, "ellipse")

            if t_end is None:
                print(f"  {desc}: 推理失败")
                continue

            d_res = abs(float(t_res) - float(b_res))
            d_end = abs(float(t_end) - float(b_end))
            agg["ellipse"].append(d_res)

            yolo_used = "✓" if debug.get("yolo_center") is not None else "✗"
            print(f"  {desc}: Δres={d_res:.4f}, YOLO中心={yolo_used}")

            # 保存调试图
            debug_name = f"{i+1:02d}_{name.split('.')[0]}_{scenario_name}.jpg"
            save_debug_comparison(name, test_img, debug,
                                 os.path.join(DEBUG_DIR, debug_name))

    print(f"\n{'='*80}")
    print(f"统计结果（基线=正面off）")
    print(f"{'='*80}")

    if agg["ellipse"]:
        vals = agg["ellipse"]
        print(f"椭圆+YOLO中心: N={len(vals)}, "
              f"mean={np.mean(vals):.4f}, "
              f"max={np.max(vals):.4f}, "
              f"std={np.std(vals):.4f}")

    print(f"\n调试图已保存至: {DEBUG_DIR}")
    print("蓝色点 = YOLO检测的中心点")
    print("绿色椭圆 = 最终使用的椭圆（轴线来自轮廓拟合，中心可能被YOLO替换）")


if __name__ == "__main__":
    main()
