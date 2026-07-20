# -*- coding: utf-8 -*-
"""
椭圆自动矫正原型 + 对照评估（绕过 FastAPI，直连 meterZeroShot）

思路：表盘是正圆，拍歪后在图里是椭圆。在表盘 crop 上拟合外圈椭圆，
用仿射把椭圆"压回"正圆（去前缩），从而不需要人工给俯仰角。

评估：对正表盘合成俯仰，分别比较
  pitch(off)        — 无矫正（复现上次的漂移）
  pitch(ellipse)    — 椭圆自动矫正
  frontal(ellipse)  — 对正表盘做矫正（确认不会把好图搞坏）
全部以"正表盘 off"为基线，量 endNum / resultNum 漂移。
"""
import os, sys, math, csv
import numpy as np
import cv2
import torch
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ)
from utils.angleDetect.zeroShotMeter import meterZeroShot

W = os.path.join(PROJ, "utils", "angleDetect")
u2net  = os.path.join(W, "pointerSeg", "resultSeg", "best.pt")
yolo   = os.path.join(W, "yoloDetection", "result", "yolo_findMeter.pt")
mclip  = os.path.join(W, "vitTranforms", "result", "best.pt")
points = os.path.join(W, "yoloDetection", "result", "yolo_pointbest.pt")
SCALE_START, SCALE_END = 0, 1.6
DEBUG_DIR = os.path.join(PROJ, "test_results", "ellipse_debug")


# ---------------- 合成俯仰（与上次实验一致） ----------------
def warp_pitch(img, phi_deg, axis='x', f_ratio=1.0, d_ratio=2.0):
    h, w = img.shape[:2]
    f, d = w * f_ratio, w * d_ratio
    phi = math.radians(phi_deg)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts = np.float32([[-w/2, -h/2, 0], [w/2, -h/2, 0], [w/2, h/2, 0], [-w/2, h/2, 0]])
    c, s = math.cos(phi), math.sin(phi)
    R = np.float32([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == 'x' \
        else np.float32([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    r = (R @ pts.T).T
    r[:, 2] += d
    dst = np.zeros((4, 2), np.float32)
    dst[:, 0] = f * r[:, 0] / r[:, 2] + w/2
    dst[:, 1] = f * r[:, 1] / r[:, 2] + h/2
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h), borderValue=(0, 0, 0))


# ---------------- 椭圆拟合 + 矫正 ----------------
def fit_dial_ellipse(crop_bgr):
    """在表盘 crop 上拟合外圈椭圆，返回 ((cx,cy),(MA,ma),angle) 或 None。"""
    h, w = crop_bgr.shape[:2]
    area_img = h * w
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    cands = []
    # 策略1: Canny 边缘 + 闭运算
    med = float(np.median(gray))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    # 策略2: Otsu 阈值（表盘面常是亮盘）
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for mask in (edges, th, cv2.bitwise_not(th)):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 20:
                continue
            try:
                e = cv2.fitEllipse(c)
            except cv2.error:
                continue
            (cx, cy), (MA, ma), ang = e
            if MA <= 1 or ma <= 1:
                continue
            ratio = min(MA, ma) / max(MA, ma)
            e_area = math.pi * (MA / 2) * (ma / 2)
            fill = e_area / area_img
            centric = math.hypot(cx - w/2, cy - h/2) / math.hypot(w/2, h/2)
            # 约束: 占图 20%~110%、不太扁、较居中
            if fill < 0.20 or fill > 1.10 or ratio < 0.30 or centric > 0.45:
                continue
            score = fill * (1 - centric) * (0.5 + 0.5 * ratio)
            cands.append((score, e))
    if not cands:
        return None
    return max(cands, key=lambda x: x[0])[1]


def affine_ellipse_to_circle(ellipse, shape):
    """构造把椭圆压回正圆（半径取长半轴）的 2x3 仿射，绕图像中心。"""
    (cx, cy), (MA, ma), ang = ellipse
    a, b = max(MA, ma) / 2.0, min(MA, ma) / 2.0
    if b < 1e-3:
        return None
    # fitEllipse 的 angle 是次轴方向，长轴方向 = ang(以 MA 为准时需判断)；统一按主轴对齐
    # 这里把椭圆主轴旋到水平，沿短轴放大 a/b，再转回，绕表盘中心(cx,cy)
    theta = math.radians(ang)
    R = np.array([[math.cos(theta), math.sin(theta)],
                  [-math.sin(theta), math.cos(theta)]])
    # MA 是第一个轴(竖直方向半径 MA/2)，ma 第二个；放大较短的那个轴
    if MA >= ma:
        S = np.diag([a / b, 1.0])   # x(短轴)放大
    else:
        S = np.diag([1.0, a / b])   # y(短轴)放大
    Ain = R.T @ S @ R               # 2x2 线性部分
    center = np.array([cx, cy])
    t = center - Ain @ center
    M = np.hstack([Ain, t.reshape(2, 1)]).astype(np.float32)
    return M


def ellipse_rectify(crop_bgr, return_debug=False):
    e = fit_dial_ellipse(crop_bgr)
    if e is None:
        return (crop_bgr, None) if return_debug else None
    M = affine_ellipse_to_circle(e, crop_bgr.shape)
    if M is None:
        return (crop_bgr, None) if return_debug else None
    h, w = crop_bgr.shape[:2]
    rect = cv2.warpAffine(crop_bgr, M, (w, h), borderValue=(0, 0, 0))
    if return_debug:
        overlay = crop_bgr.copy()
        cv2.ellipse(overlay, e, (0, 255, 0), 2)
        return rect, overlay
    return rect


# ---------------- 子类: 新增 ellipse correction_mode + 调试导出 ----------------
class MeterEllipse(meterZeroShot):
    _dbg_left = 0

    def _correct_meter_image(self, corpImg, correction_mode, confidence=None,
                             start_end_position="start_left_end_right",
                             stretch_x_ratio=1.0, stretch_y_ratio=1.0):
        if str(correction_mode).strip().lower() == "ellipse":
            try:
                rect, overlay = ellipse_rectify(corpImg, return_debug=True)
            except Exception as e:
                logger.warning(f"ellipse rectify failed: {e}")
                return corpImg
            if MeterEllipse._dbg_left > 0 and overlay is not None:
                MeterEllipse._dbg_left -= 1
                os.makedirs(DEBUG_DIR, exist_ok=True)
                hh = 240
                def fit(im):
                    return cv2.resize(im, (int(im.shape[1]*hh/im.shape[0]), hh))
                mont = np.hstack([fit(corpImg), fit(overlay), fit(rect)])
                idx = MeterEllipse._dbg_left
                cv2.imwrite(os.path.join(DEBUG_DIR, f"dbg_{idx}.jpg"), mont)
            return rect if rect is not None else corpImg
        return super()._correct_meter_image(
            corpImg, correction_mode, confidence=confidence,
            start_end_position=start_end_position,
            stretch_x_ratio=stretch_x_ratio, stretch_y_ratio=stretch_y_ratio)


def run(model, img, mode="off"):
    out = model.Inference(img, SCALE_START, SCALE_END, correction_mode=mode)
    return out[0], out[1]  # endNum, resultNum


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=", device)
    model = MeterEllipse(u2net, yolo, mclip, points, device)
    MeterEllipse._dbg_left = 6  # 导出前6张 ellipse 调试拼图

    data_dir = os.path.join(PROJ, "data")
    imgs = sorted([f for f in os.listdir(data_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    conds = ["pitch30_off", "pitch30_ellipse",
             "pitch45_off", "pitch45_ellipse",
             "frontal_ellipse"]
    agg = {c: [] for c in conds}
    fail = {c: 0 for c in conds}

    for i, name in enumerate(imgs):
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None:
            continue
        b_end, b_res = run(model, img, "off")
        if b_end is None:
            print(f"[{i+1}/{len(imgs)}] {name} baseline FAIL")
            continue
        p30 = warp_pitch(img, 30)
        p45 = warp_pitch(img, 45)
        samples = {
            "pitch30_off":     (p30, "off"),
            "pitch30_ellipse": (p30, "ellipse"),
            "pitch45_off":     (p45, "off"),
            "pitch45_ellipse": (p45, "ellipse"),
            "frontal_ellipse": (img, "ellipse"),
        }
        line = f"[{i+1}/{len(imgs)}] {name} base={b_res:.3f} | "
        for c in conds:
            im, mode = samples[c]
            try:
                t_end, t_res = run(model, im, mode)
            except Exception:
                t_end = None
            if t_end is None:
                fail[c] += 1
                line += f"{c.split('_')[1][:4]}=FAIL "
                continue
            d = abs(float(t_res) - float(b_res))
            de = abs(float(t_end) - float(b_end))
            agg[c].append((de, d))
            line += f"{c.split('_')[1][:4]}={d:.3f} "
        print(line)

    print("\n========= 椭圆自动矫正评估（基线=正表盘 off）=========")
    print(f"{'condition':<20}{'N':>4}{'fail':>6}{'mean|dEnd|':>12}{'mean|dRes|':>12}{'max|dRes|':>11}")
    for c in conds:
        v = agg[c]
        if v:
            mE = sum(x[0] for x in v)/len(v)
            mR = sum(x[1] for x in v)/len(v)
            xR = max(x[1] for x in v)
            print(f"{c:<20}{len(v):>4}{fail[c]:>6}{mE:>12.2f}{mR:>12.4f}{xR:>11.4f}")
        else:
            print(f"{c:<20}{0:>4}{fail[c]:>6}{'-':>12}{'-':>12}{'-':>11}")
    print(f"\n调试拼图(原crop|椭圆拟合|矫正后): {DEBUG_DIR}")
    print("对照参考(上次实验): pitch30 off mean|dRes|=0.081 fail=5 ; 已知角矫正=0.011 fail=0")


if __name__ == "__main__":
    main()
