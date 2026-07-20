# -*- coding: utf-8 -*-
"""
椭圆自动矫正 v2：先锁定"白色表盘面"再拟椭圆（不需人工俯仰角）。

v1 失败根因：Canny/Otsu 抓到外圈金属壳，圈大偏心 -> 仿射过矫正。
v2 改进：
  1. 用亮度+低饱和分割出中央白色表盘面，取含中心的最大连通域，对其轮廓 fitEllipse
  2. 仿射矫正写干净：T(c) R Sdiag R^T T(-c)，把椭圆压成正圆(半径取长半轴)
  3. 拟合质量校验(轴比/占比/居中/中心偏移)不过就回退 off
重点先核对调试图：椭圆是否贴住表盘面。
"""
import os, sys, math
import numpy as np
import cv2
import torch
from loguru import logger

logger.remove(); logger.add(sys.stderr, level="WARNING")
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ)
from utils.angleDetect.zeroShotMeter import meterZeroShot

W = os.path.join(PROJ, "utils", "angleDetect")
u2net  = os.path.join(W, "pointerSeg", "resultSeg", "best.pt")
yolo   = os.path.join(W, "yoloDetection", "result", "yolo_findMeter.pt")
mclip  = os.path.join(W, "vitTranforms", "result", "best.pt")
points = os.path.join(W, "yoloDetection", "result", "yolo_pointbest.pt")
SCALE_START, SCALE_END = 0, 1.6
DEBUG_DIR = os.path.join(PROJ, "test_results", "ellipse_debug_v2")


def warp_pitch(img, phi_deg, axis='x', f_ratio=1.0, d_ratio=2.0):
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


def fit_face_ellipse(crop_bgr):
    """锁定中央白色表盘面拟椭圆。返回 ellipse 或 None。"""
    h, w = crop_bgr.shape[:2]
    area_img = h * w
    cx0, cy0 = w/2, h/2
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H_, S_, V_ = cv2.split(hsv)
    # 白色面: 高亮度 + 低饱和
    v_thr = np.percentile(V_, 55)
    mask = ((V_ >= v_thr) & (S_ <= 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8), iterations=2)

    num, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    if num < 2:
        return None
    # 选: 含中心 或 离中心最近的大连通域
    best, best_score = None, -1
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.10 * area_img:
            continue
        ccx, ccy = cents[i]
        centric = math.hypot(ccx-cx0, ccy-cy0) / math.hypot(cx0, cy0)
        if centric > 0.40:
            continue
        score = area * (1 - centric)
        if score > best_score:
            best_score, best = score, i
    if best is None:
        return None
    comp = (lab == best).astype(np.uint8) * 255
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 20:
        return None
    e = cv2.fitEllipse(c)
    (ecx, ecy), (d1, d2), ang = e
    a, b = max(d1, d2)/2, min(d1, d2)/2
    if b < 1:
        return None
    ratio = b / a
    fill = math.pi * a * b / area_img
    centric = math.hypot(ecx-cx0, ecy-cy0) / math.hypot(cx0, cy0)
    # 质量校验: 不太扁、占比合理、较居中
    if ratio < 0.45 or fill < 0.12 or fill > 1.0 or centric > 0.35:
        return None
    return e


def rectify_affine(crop_bgr, e):
    (cx, cy), (d1, d2), ang = e
    a1, a2 = d1/2.0, d2/2.0
    target = max(a1, a2)
    th = math.radians(ang)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    S = np.diag([target/a1, target/a2])
    A = R @ S @ R.T
    c = np.array([cx, cy])
    t = c - A @ c
    M = np.hstack([A, t.reshape(2,1)]).astype(np.float32)
    h, w = crop_bgr.shape[:2]
    return cv2.warpAffine(crop_bgr, M, (w, h), borderValue=(0,0,0))


class MeterE2(meterZeroShot):
    _dbg = 0
    _found = 0
    _calls = 0
    def _correct_meter_image(self, corpImg, correction_mode, confidence=None,
                             start_end_position="start_left_end_right",
                             stretch_x_ratio=1.0, stretch_y_ratio=1.0):
        if str(correction_mode).strip().lower() == "ellipse2":
            MeterE2._calls += 1
            try:
                e = fit_face_ellipse(corpImg)
            except Exception as ex:
                logger.warning(f"fit fail {ex}"); e = None
            if e is None:
                return corpImg
            MeterE2._found += 1
            rect = rectify_affine(corpImg, e)
            if MeterE2._dbg > 0:
                MeterE2._dbg -= 1
                os.makedirs(DEBUG_DIR, exist_ok=True)
                ov = corpImg.copy(); cv2.ellipse(ov, e, (0,255,0), 2)
                hh = 240
                f = lambda im: cv2.resize(im, (int(im.shape[1]*hh/im.shape[0]), hh))
                cv2.imwrite(os.path.join(DEBUG_DIR, f"dbg_{MeterE2._dbg}.jpg"),
                            np.hstack([f(corpImg), f(ov), f(rect)]))
            return rect
        return super()._correct_meter_image(corpImg, correction_mode, confidence=confidence,
            start_end_position=start_end_position, stretch_x_ratio=stretch_x_ratio,
            stretch_y_ratio=stretch_y_ratio)


def run(model, img, mode="off"):
    out = model.Inference(img, SCALE_START, SCALE_END, correction_mode=mode)
    return out[0], out[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=", device)
    model = MeterE2(u2net, yolo, mclip, points, device)
    MeterE2._dbg = 8

    data_dir = os.path.join(PROJ, "data")
    imgs = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".jpg",".png",".jpeg"))])
    conds = ["pitch30_off","pitch30_e2","pitch45_off","pitch45_e2","frontal_e2"]
    agg = {c: [] for c in conds}; fail = {c: 0 for c in conds}

    for i, name in enumerate(imgs):
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None: continue
        b_end, b_res = run(model, img, "off")
        if b_end is None:
            print(f"[{i+1}] {name} base FAIL"); continue
        p30, p45 = warp_pitch(img,30), warp_pitch(img,45)
        samples = {"pitch30_off":(p30,"off"),"pitch30_e2":(p30,"ellipse2"),
                   "pitch45_off":(p45,"off"),"pitch45_e2":(p45,"ellipse2"),
                   "frontal_e2":(img,"ellipse2")}
        line = f"[{i+1}] base={b_res:.3f} | "
        for c in conds:
            im, mode = samples[c]
            try: t_end, t_res = run(model, im, mode)
            except Exception: t_end = None
            if t_end is None:
                fail[c]+=1; line += f"{c.split('_')[1]}=FAIL "; continue
            d = abs(float(t_res)-float(b_res)); de = abs(float(t_end)-float(b_end))
            agg[c].append((de,d)); line += f"{c.split('_')[1]}={d:.3f} "
        print(line)

    print(f"\n表盘面椭圆命中率: {MeterE2._found}/{MeterE2._calls}")
    print("========= v2 评估（基线=正表盘 off）=========")
    print(f"{'condition':<16}{'N':>4}{'fail':>6}{'mean|dEnd|':>12}{'mean|dRes|':>12}{'max|dRes|':>11}")
    for c in conds:
        v = agg[c]
        if v:
            mE=sum(x[0] for x in v)/len(v); mR=sum(x[1] for x in v)/len(v); xR=max(x[1] for x in v)
            print(f"{c:<16}{len(v):>4}{fail[c]:>6}{mE:>12.2f}{mR:>12.4f}{xR:>11.4f}")
        else:
            print(f"{c:<16}{0:>4}{fail[c]:>6}{'-':>12}{'-':>12}{'-':>11}")
    print(f"\n调试图: {DEBUG_DIR}")
    print("对照: v1 pitch30_ellipse=0.121 ; pitch30_off=0.081 ; 已知角=0.011")


if __name__ == "__main__":
    main()
