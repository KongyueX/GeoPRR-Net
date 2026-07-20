# -*- coding: utf-8 -*-
"""
角度敏感性对照实验（直连 meterZeroShot，绕过 FastAPI）

目的：量化"拍摄角度不正"对最终读数精度的影响，并验证"已知俯仰角后做反透视矫正"能救回多少。

对每张正表盘图，以正表盘推理结果为基线(baseline)，分别施加：
  - 平面内旋转 rot±deg
  - 俯仰透视 pitch（绕水平X轴旋转表盘平面的 homography）
  - pitch + 已知角度反矫正 rectified
统计 endNum(transformer 0-100) 和 resultNum(最终读数) 相对基线的漂移。

注意：data/ 是正表盘、无真值读数，所以这里测的是"相对正表盘基线的漂移"，
不是绝对精度——正表盘基线本身被当作参考真值。
"""
import os, sys, math, csv
import numpy as np
import cv2
import torch
from loguru import logger

logger.remove()  # 静默 zeroShotMeter 的海量 INFO 日志
logger.add(sys.stderr, level="WARNING")

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ)
from utils.angleDetect.zeroShotMeter import meterZeroShot

W = os.path.join(PROJ, "utils", "angleDetect")
u2net   = os.path.join(W, "pointerSeg", "resultSeg", "best.pt")
yolo    = os.path.join(W, "yoloDetection", "result", "yolo_findMeter.pt")
mclip   = os.path.join(W, "vitTranforms", "result", "best.pt")
points  = os.path.join(W, "yoloDetection", "result", "yolo_pointbest.pt")

SCALE_START, SCALE_END = 0, 1.6


def rotate_inplane(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))


def pitch_homography(img_shape, phi_deg, axis='x', f_ratio=1.0, d_ratio=2.0):
    """构造把正表盘平面绕轴旋转 phi 度后的透视 homography。axis='x' 为俯仰。"""
    h, w = img_shape[:2]
    f = w * f_ratio
    d = w * d_ratio
    phi = math.radians(phi_deg)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0],
                      [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    c, s = math.cos(phi), math.sin(phi)
    if axis == 'x':
        R = np.float32([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.float32([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    r = (R @ pts.T).T
    r[:, 2] += d
    dst = np.zeros((4, 2), np.float32)
    dst[:, 0] = f * r[:, 0] / r[:, 2] + w / 2
    dst[:, 1] = f * r[:, 1] / r[:, 2] + h / 2
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def warp_pitch(img, phi_deg, axis='x'):
    H = pitch_homography(img.shape, phi_deg, axis)
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), borderValue=(0, 0, 0)), H


def warp_pitch_then_rectify(img, phi_deg, axis='x'):
    """先施加俯仰透视，再用已知角度的反 homography 矫正回去（模拟'给出俯仰角'）。"""
    warped, H = warp_pitch(img, phi_deg, axis)
    h, w = img.shape[:2]
    rect = cv2.warpPerspective(warped, np.linalg.inv(H), (w, h), borderValue=(0, 0, 0))
    return rect


def run(model, img):
    endNum, resultNum, *_ = model.Inference(img, SCALE_START, SCALE_END,
                                            correction_mode="off")  # 关闭内置矫正，单独看角度影响
    return endNum, resultNum


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model = meterZeroShot(u2net, yolo, mclip, points, device)

    data_dir = os.path.join(PROJ, "data")
    imgs = sorted([f for f in os.listdir(data_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"images={len(imgs)}")

    conditions = [
        ("rot+15",   lambda im: rotate_inplane(im, 15)),
        ("rot-15",   lambda im: rotate_inplane(im, -15)),
        ("rot+30",   lambda im: rotate_inplane(im, 30)),
        ("pitch15",  lambda im: warp_pitch(im, 15)[0]),
        ("pitch30",  lambda im: warp_pitch(im, 30)[0]),
        ("pitch45",  lambda im: warp_pitch(im, 45)[0]),
        ("pitch30_rectified", lambda im: warp_pitch_then_rectify(im, 30)),
    ]

    rows = []
    # 累加器： cond -> list of (d_end, d_result)
    agg = {c[0]: [] for c in conditions}
    fail = {c[0]: 0 for c in conditions}
    base_fail = 0

    for i, name in enumerate(imgs):
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None:
            continue
        b_end, b_res = run(model, img)
        if b_end is None:
            base_fail += 1
            print(f"[{i+1}/{len(imgs)}] {name}  baseline FAIL(未检测到表盘/指针)")
            rows.append([name, "baseline", "", "", "FAIL"])
            continue
        rows.append([name, "baseline", b_end, f"{b_res:.4f}", ""])
        line = f"[{i+1}/{len(imgs)}] {name}  base end={b_end} res={b_res:.3f} | "
        for cname, fn in conditions:
            try:
                t_end, t_res = run(model, fn(img))
            except Exception as e:
                t_end = None
            if t_end is None:
                fail[cname] += 1
                rows.append([name, cname, "", "", "FAIL"])
                line += f"{cname}=FAIL "
                continue
            d_end = abs(float(t_end) - float(b_end))
            d_res = abs(float(t_res) - float(b_res))
            agg[cname].append((d_end, d_res))
            rows.append([name, cname, t_end, f"{t_res:.4f}",
                         f"dEnd={d_end:.1f} dRes={d_res:.4f}"])
            line += f"{cname}=d{d_res:.3f} "
        print(line)

    # 写明细 CSV
    out_csv = os.path.join(PROJ, "test_results", "angle_sensitivity.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "condition", "endNum", "resultNum", "note"])
        w.writerows(rows)

    # 汇总
    print("\n================ 汇总（相对正表盘基线的平均漂移）================")
    print(f"{'condition':<20}{'N':>4}{'fail':>6}{'mean|dEnd|':>12}{'mean|dRes|':>12}{'max|dRes|':>12}")
    print(f"{'baseline':<20}{'-':>4}{base_fail:>6}{'-':>12}{'-':>12}{'-':>12}")
    for cname, _ in conditions:
        vals = agg[cname]
        if vals:
            mEnd = sum(v[0] for v in vals) / len(vals)
            mRes = sum(v[1] for v in vals) / len(vals)
            xRes = max(v[1] for v in vals)
            print(f"{cname:<20}{len(vals):>4}{fail[cname]:>6}{mEnd:>12.2f}{mRes:>12.4f}{xRes:>12.4f}")
        else:
            print(f"{cname:<20}{0:>4}{fail[cname]:>6}{'-':>12}{'-':>12}{'-':>12}")
    print(f"\n明细已写入: {out_csv}")
    print("说明: 量程", SCALE_START, "~", SCALE_END,
          " | mean|dRes| 是最终读数相对正表盘基线的平均绝对漂移")


if __name__ == "__main__":
    main()
