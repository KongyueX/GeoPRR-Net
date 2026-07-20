# -*- coding: utf-8 -*-
"""
用真值验证椭圆校正策略准确性（含YOLO中心点约束）v2

直接复用生产代码的完整ellipse矫正流程，只替换mask生成部分
生成HTML可视化报告

运行: D:/minconda/envs/pytorch5060/python.exe verify_with_yolo_center_v2.py
"""
import os, sys
import cv2
import numpy as np
import math
import torch
from loguru import logger
import base64

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


GROUND_TRUTH = {
    "13.png": 0.17,
    "234.png": 0.17,
    "253.png": None,
    "345.png": 0.175,
    "23423.png": 0.185,
    "32333.png": 0.46,
}


def img_to_base64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('ascii')


# ==================== mask策略 ====================

def mask_hsv_baseline(crop_bgr):
    """当前HSV方案"""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    _, S_, V_ = cv2.split(hsv)
    v_thr = np.percentile(V_, 55)
    mask = ((V_ >= v_thr) & (S_ <= 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    return mask


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
    return mask_fused


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
    return mask


# ==================== 核心：用指定mask策略 + YOLO中心拟合椭圆 ====================

def fit_ellipse_from_mask_with_yolo(mask, crop_bgr, model):
    """
    从mask提取椭圆，集成YOLO中心点替换逻辑
    返回 (ellipse, comp, yolo_center, yolo_used, original_center, status)
    """
    h, w = crop_bgr.shape[:2]
    area_img = float(h * w)
    cx0, cy0 = w / 2.0, h / 2.0

    num, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    if num < 2:
        return None, None, None, False, None, "No components"

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
        return None, None, None, False, None, "No valid component"

    comp = (lab == best).astype(np.uint8) * 255
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None, False, None, "No contours"

    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20:
        return None, None, None, False, None, "Too few points"

    ellipse = cv2.fitEllipse(c)
    (ecx, ecy), (d1, d2), ang = ellipse
    a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0
    if b < 1:
        return None, None, None, False, None, "Invalid axes"

    ratio = b / a
    fill = math.pi * a * b / area_img
    centric = math.hypot(ecx - cx0, ecy - cy0) / math.hypot(cx0, cy0)

    if ratio < 0.45 or fill < 0.12 or fill > 1.0 or centric > 0.35:
        reason = f"Quality fail (r={ratio:.2f},f={fill:.2f},c={centric:.2f})"
        return None, None, None, False, None, reason

    # 尝试YOLO中心点替换（复用生产代码逻辑）
    yolo_center = None
    yolo_used = False
    original_center = (ecx, ecy)

    try:
        yolo_center, _ = model.pointerDetect.center_find(crop_bgr, classId=0)
        if yolo_center is not None:
            yolo_dist = math.hypot(yolo_center[0] - cx0, yolo_center[1] - cy0)
            max_dist = math.hypot(cx0, cy0) * 0.5
            if yolo_dist <= max_dist:
                ellipse = (yolo_center, (d1, d2), ang)
                yolo_used = True
    except Exception:
        pass

    return ellipse, comp, yolo_center, yolo_used, original_center, "OK"


def rectify_ellipse_affine(crop_bgr, ellipse):
    """仿射变换压成正圆（复用生产代码）"""
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


def draw_ellipse_debug(crop_bgr, ellipse, yolo_center, original_center, yolo_used):
    """绘制椭圆+中心点可视化"""
    vis = crop_bgr.copy()
    cv2.ellipse(vis, ellipse, (0, 255, 0), 2)

    # 拟合中心（红色）
    if original_center is not None:
        cv2.circle(vis, (int(original_center[0]), int(original_center[1])), 5, (0, 0, 255), -1)
        cv2.putText(vis, "Fit", (int(original_center[0])+8, int(original_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # YOLO中心（蓝色）
    if yolo_center is not None:
        cv2.circle(vis, (int(yolo_center[0]), int(yolo_center[1])), 5, (255, 0, 0), -1)
        cv2.putText(vis, "YOLO", (int(yolo_center[0])+8, int(yolo_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 最终使用的中心（绿色外圈）
    (fx, fy), _, _ = ellipse
    cv2.circle(vis, (int(fx), int(fy)), 7, (0, 255, 0), 2)

    status = "YOLO✓" if yolo_used else "Fit"
    cv2.putText(vis, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return vis


def custom_inference_with_ellipse(model, img, strategy_fn, scale_start, scale_end):
    """
    自定义推理：用指定策略生成mask，集成YOLO中心，手动执行椭圆矫正
    返回 (reading, crop, mask, ellipse_vis, rectified, yolo_used, status)
    """
    # 1. YOLO检测表盘裁剪
    try:
        _, _, all_crops, _, best_idx = model.meterDetect.image_crop(img, size=0)
        if not all_crops:
            return None, None, None, None, None, False, "No meter detected"
        crop = all_crops[best_idx]
    except Exception as e:
        return None, None, None, None, None, False, f"Crop failed: {e}"

    # 2. 用指定策略生成mask
    try:
        mask = strategy_fn(crop)
    except Exception as e:
        return None, crop, None, None, None, False, f"Mask failed: {e}"

    # 3. 拟合椭圆 + YOLO中心替换
    ellipse, comp, yolo_center, yolo_used, orig_center, status = fit_ellipse_from_mask_with_yolo(mask, crop, model)

    if ellipse is None:
        return None, crop, mask, None, None, False, status

    # 4. 绘制调试可视化
    ellipse_vis = draw_ellipse_debug(crop, ellipse, yolo_center, orig_center, yolo_used)

    # 5. 检查是否需要矫正（已经够圆就跳过）
    (_, _), (d1, d2), _ = ellipse
    ratio = min(d1, d2) / max(d1, d2)
    if ratio >= 0.95:
        rectified = crop.copy()
    else:
        rectified = rectify_ellipse_affine(crop, ellipse)

    # 6. 继续后续推理（指针分割+transformer+刻度检测）
    try:
        # 暂时替换model的_ellipse_rectify，让它直接返回我们的矫正结果
        original_rectify = model._ellipse_rectify
        model._ellipse_rectify = lambda x: rectified

        _, reading, _, _, _ = model.Inference(img, scale_start, scale_end, correction_mode="ellipse")

        model._ellipse_rectify = original_rectify

        return reading, crop, mask, ellipse_vis, rectified, yolo_used, "OK"

    except Exception as e:
        model._ellipse_rectify = original_rectify
        return None, crop, mask, ellipse_vis, rectified, yolo_used, f"Inference failed: {e}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device= {device}\n")

    model = meterZeroShot(device=device, **WEIGHTS)

    data_dir = os.path.join(PROJ, "data")
    strategies = [
        ("off", None),
        ("HSV", mask_hsv_baseline),
        ("Fusion", mask_hsv_edge_fusion),
        ("Edge", mask_edge_direct),
    ]

    results = []

    for name, truth in GROUND_TRUTH.items():
        if truth is None:
            print(f"[跳过] {name} - 过曝无真值")
            continue

        img_path = os.path.join(data_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[错误] {name} - 读取失败")
            continue

        row = {"name": name, "truth": truth, "original": img_to_base64(img)}

        for label, strategy in strategies:
            if label == "off":
                # off基线：直接用生产代码
                _, res, _, _, _ = model.Inference(img, SCALE_START, SCALE_END, correction_mode="off")
                row[label] = {
                    "reading": res,
                    "error": abs(float(res) - truth) if res is not None else None,
                    "mask": None,
                    "ellipse_vis": None,
                    "rectified": None,
                    "yolo_used": False,
                    "status": "OK" if res is not None else "Failed",
                }
            else:
                # 用指定策略 + YOLO中心
                res, crop, mask, ellipse_vis, rectified, yolo_used, status = custom_inference_with_ellipse(
                    model, img, strategy, SCALE_START, SCALE_END
                )

                row[label] = {
                    "reading": res,
                    "error": abs(float(res) - truth) if res is not None else None,
                    "mask": img_to_base64(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)) if mask is not None else None,
                    "ellipse_vis": img_to_base64(ellipse_vis) if ellipse_vis is not None else None,
                    "rectified": img_to_base64(rectified) if rectified is not None else None,
                    "yolo_used": yolo_used,
                    "status": status,
                }

        results.append(row)

        # 打印这行结果
        print(f"{name:<20} 真值={truth:.3f}")
        for label, _ in strategies:
            data = row[label]
            if data["reading"] is not None:
                yolo_mark = " [YOLO✓]" if data.get("yolo_used") else ""
                print(f"  {label:<10} 读数={data['reading']:.4f}  误差={data['error']:.4f}{yolo_mark}")
            else:
                print(f"  {label:<10} 失败 - {data.get('status', '未知')}")
        print()

    # 汇总统计
    print("=" * 70)
    print("汇总统计（平均绝对误差）")
    print("=" * 70)

    summary = {}
    for label, _ in strategies:
        errors = [row[label]["error"] for row in results if row[label]["error"] is not None]
        yolo_count = sum(1 for row in results if row[label].get("yolo_used"))
        if errors:
            mean_err = sum(errors) / len(errors)
            max_err = max(errors)
            summary[label] = (len(errors), mean_err, max_err, yolo_count)
        else:
            summary[label] = (0, None, None, 0)

    for label, (n, mean_err, max_err, yolo_cnt) in summary.items():
        if mean_err is not None:
            yolo_info = f"  YOLO命中={yolo_cnt}/{n}" if label != "off" else ""
            print(f"{label:<15} N={n}  平均误差={mean_err:.4f}  最大误差={max_err:.4f}{yolo_info}")
        else:
            print(f"{label:<15} N={n}  无有效数据")

    print(f"\n✓ 生成HTML报告...")
    generate_html_report(results, summary, strategies)


def generate_html_report(results, summary, strategies):
    """生成HTML可视化报告"""
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>椭圆校正真值验证（含YOLO中心点）</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:#0a0c10;color:#e6e6e6;font:14px/1.5 sans-serif;padding:20px}}
h1{{font-size:22px;margin:0 0 8px;color:#7fd1ff}}
.meta{{color:#9aa4b2;font-size:13px;margin-bottom:20px}}
.summary{{border-collapse:collapse;margin-bottom:24px;background:#141821;border-radius:8px;overflow:hidden;width:100%}}
.summary th,.summary td{{padding:8px 16px;text-align:center;border:1px solid #232733;font-size:13px}}
.summary th{{background:#1a1f2a;color:#7fd1ff}}
.summary td:first-child{{text-align:left;color:#cbd5e1}}
.card{{background:#141821;border:1px solid #232733;border-radius:10px;padding:16px;margin-bottom:16px}}
.card h3{{font-size:15px;margin:0 0 12px;color:#cbd5e1}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:12px}}
figure{{margin:0;text-align:center}}
figure img{{max-width:100%;height:auto;border-radius:6px;border:1px solid #2a2f3a;background:#000}}
figure.fail{{opacity:0.5}}
figcaption{{font-size:11px;color:#7a8493;margin-top:4px}}
.ok{{color:#5ee08a}}
.fail{{color:#ff8b8b}}
.warn{{color:#ffcf6a}}
.yolo{{color:#7fd1ff}}
</style></head>
<body>
<h1>椭圆校正真值验证（含YOLO中心点约束）</h1>
<div class="meta">对比 off / HSV / HSV+Edge Fusion / Edge Direct 四种方案，集成YOLO检测的表盘中心点替换拟合中心。<br>
<strong>可视化说明</strong>：红点=拟合中心，蓝点=YOLO中心，绿圈=最终使用的中心。</div>

<table class="summary">
<tr><th>策略</th><th>样本数</th><th>平均误差</th><th>最大误差</th><th>YOLO命中</th></tr>
"""

    for label, _ in strategies:
        n, mean_err, max_err, yolo_cnt = summary[label]
        if mean_err is not None:
            yolo_info = f"{yolo_cnt}/{n}" if label != "off" else "—"
            html += f"""<tr><td>{label}</td><td>{n}</td><td>{mean_err:.4f}</td><td>{max_err:.4f}</td><td class="yolo">{yolo_info}</td></tr>\n"""
        else:
            html += f"""<tr><td>{label}</td><td>0</td><td>—</td><td>—</td><td>—</td></tr>\n"""

    html += "</table>\n"

    # 每张图的详细对比
    for row in results:
        html += f"""<div class="card">
<h3>{row['name']} <span style="color:#5ee08a">真值={row['truth']:.3f}</span></h3>
<div class="grid">
<figure><img src="data:image/png;base64,{row['original']}"><figcaption>原图</figcaption></figure>
"""

        for label, _ in strategies:
            data = row[label]
            fail_class = ' class="fail"' if data["reading"] is None else ""

            if data["reading"] is not None:
                error_class = "ok" if data["error"] < 0.05 else ("warn" if data["error"] < 0.10 else "fail")
                yolo_mark = ' <span class="yolo">[YOLO✓]</span>' if data.get("yolo_used") else ""

                html += f"""<figure{fail_class}>
"""
                if data.get("ellipse_vis"):
                    html += f"""<img src="data:image/png;base64,{data['ellipse_vis']}">"""
                elif label == "off":
                    html += f"""<div style="height:120px;display:flex;align-items:center;justify-content:center;color:#9aa4b2;border:1px solid #232733;border-radius:6px">无矫正</div>"""

                html += f"""<figcaption>{label}: <span class="{error_class}">{data['reading']:.4f}</span> (误差={data['error']:.4f}){yolo_mark}</figcaption></figure>
"""
            else:
                html += f"""<figure class="fail"><div style="height:120px;display:flex;align-items:center;justify-content:center;color:#5a6472;border:1px dashed #2a2f3a;border-radius:6px;font-size:24px">✗</div><figcaption class="fail">{label}: {data.get('status', '失败')}</figcaption></figure>
"""

        html += "</div></div>\n"

    html += "</body></html>"

    output_path = os.path.join(PROJ, "test_results", "ellipse_yolo_center_verification.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✓ HTML报告已生成: {output_path}")


if __name__ == "__main__":
    main()
