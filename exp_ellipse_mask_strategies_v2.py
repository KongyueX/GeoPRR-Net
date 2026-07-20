# -*- coding: utf-8 -*-
"""
椭圆校正mask生成方案对比实验 v2 - 拆分正拍/斜拍统计

对比3种方案:
1. HSV当前方案 (baseline)
2. HSV + Canny边缘融合
3. Canny边缘直接拟合

关键改进: 按椭圆轴比拆分正拍(≥0.85)和斜拍(<0.85)分别统计Δ，避免平均值被正拍带偏。

运行: D:/minconda/envs/pytorch5060/python.exe exp_ellipse_mask_strategies_v2.py
"""
import os, sys, base64, html
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
THUMB_H = 180
RATIO_THRESHOLD = 0.85  # 轴比>=0.85为正拍，<0.85为斜拍


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


# ==================== 椭圆拟合与质量校验 ====================

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


def rectify_affine(crop_bgr, ellipse):
    """仿射矫正：把椭圆压成正圆"""
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


# ==================== 可视化与报告生成 ====================

def to_data_uri(img, h=THUMB_H):
    if img is None:
        return None
    im = img
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[0] != h:
        scale = h / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), h))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def draw_ellipse_overlay(crop_bgr, ellipse):
    if ellipse is None:
        return None
    out = crop_bgr.copy()
    cv2.ellipse(out, ellipse, (0, 255, 0), 2)
    return out


def figure(uri, label):
    if uri is None:
        return f'<figure class="miss"><div class="ph">×</div><figcaption>{html.escape(label)}</figcaption></figure>'
    return f'<figure><img src="{uri}"><figcaption>{html.escape(label)}</figcaption></figure>'


def fmt_delta(d):
    return f"Δ{d:.3f}" if d is not None else "Δ—"


def fmt_num(n):
    return f"{n:.4f}" if n is not None else "—"


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
    print(f"device= {device}")

    model = meterZeroShot(device=device, **WEIGHTS)

    data_dir = os.path.join(PROJ, "data")
    imgs = sorted([f for f in os.listdir(data_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))])[:20]
    print(f"Testing {len(imgs)} images (ratio threshold={RATIO_THRESHOLD})")

    strategies = [mask_hsv_baseline, mask_hsv_edge_fusion, mask_edge_direct]

    orig_fit = model._fit_face_ellipse

    cards = []
    # 统计：记录每条数据的(delta, ratio, area)
    agg = {s.__name__: {"data": [], "fail": 0} for s in strategies}

    for i, name in enumerate(imgs):
        img_path = os.path.join(data_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        o_end, o_res, _, _, _ = model.Inference(img, SCALE_START, SCALE_END,
                                                correction_mode="off")

        _, _, all_crops, _, best_idx = model.meterDetect.image_crop(img, use_origin_when_no_meter=False)
        if best_idx is None or not all_crops:
            continue
        crop = all_crops[best_idx]

        results = []
        for strategy in strategies:
            mask, label = strategy(crop)
            ellipse, comp, status = fit_ellipse_from_mask(mask, crop)

            if ellipse is not None:
                (_, _), (d1, d2), _ = ellipse
                ratio = min(d1, d2) / max(d1, d2)
                ellipse_area = math.pi * (d1 / 2) * (d2 / 2)
                area_pct = ellipse_area / (crop.shape[0] * crop.shape[1])
                rectified = rectify_affine(crop, ellipse) if ratio < 0.95 else crop
                overlay = draw_ellipse_overlay(crop, ellipse)

                model._fit_face_ellipse = make_patched_fit(strategy, fit_ellipse_from_mask)
                try:
                    e_end, e_res, _, _, _ = model.Inference(
                        img, SCALE_START, SCALE_END, correction_mode="ellipse")
                except Exception:
                    e_res = None
                model._fit_face_ellipse = orig_fit

                delta = None
                if o_res is not None and e_res is not None:
                    delta = abs(float(e_res) - float(o_res))
                    agg[strategy.__name__]["data"].append((delta, ratio, area_pct))
            else:
                rectified = None
                overlay = None
                ratio = None
                area_pct = None
                e_res = None
                delta = None
                agg[strategy.__name__]["fail"] += 1

            results.append({
                "label": label,
                "mask": mask,
                "overlay": overlay,
                "rectified": rectified,
                "ellipse": ellipse,
                "status": status,
                "ratio": ratio,
                "area_pct": area_pct,
                "e_res": e_res,
                "delta": delta,
            })

        # 生成对比卡片
        rows = []
        for r in results:
            if r['ellipse']:
                info = f"轴比={r['ratio']:.3f} 占比={r['area_pct']*100:.0f}%"
                read_txt = f"读数={r['e_res']:.4f}" if r['e_res'] is not None else "读数=—"
                delta_txt = f"Δ={r['delta']:.4f}" if r['delta'] is not None else "Δ=—"
                delta_cls = "warn" if (r['delta'] is not None and r['delta'] > 0.05) else ""
                badge = '<span class="ok">✓</span>'
                group = "frontal" if r['ratio'] >= RATIO_THRESHOLD else "tilted"
            else:
                info = r['status']
                read_txt = "读数=—"
                delta_txt = ""
                delta_cls = ""
                badge = '<span class="fail">✗</span>'
                group = ""

            rows.append(
                f'<div class="row {group}">'
                f'<div class="lbl">{badge} {html.escape(r["label"])}<br>'
                f'<small>{info}</small><br>'
                f'<small class="{delta_cls}">{read_txt} {delta_txt}</small></div>'
                f'{figure(to_data_uri(r["mask"]), "mask")}'
                f'{figure(to_data_uri(r["overlay"]), "ellipse")}'
                f'{figure(to_data_uri(r["rectified"]), "rectified")}'
                f'</div>'
            )

        o_txt = f"{o_res:.4f}" if o_res is not None else "—"
        cards.append(
            f'<section class="card">'
            f'<h3>#{i+1} {html.escape(name)} · off基线={o_txt}</h3>'
            f'<div class="grid">'
            f'{figure(to_data_uri(crop), "原始crop")}'
            f'</div>'
            f'<div class="comp">{"".join(rows)}</div>'
            f'</section>'
        )

        print(f"[{i+1}/{len(imgs)}] {name} off={o_txt} | "
              f"HSV:{results[0]['status']}({fmt_delta(results[0]['delta'])}), "
              f"Fusion:{results[1]['status']}({fmt_delta(results[1]['delta'])}), "
              f"Edge:{results[2]['status']}({fmt_delta(results[2]['delta'])})")

    # 汇总统计 - 拆分正拍/斜拍
    print("\n" + "=" * 80)
    print(f"汇总统计（轴比≥{RATIO_THRESHOLD}为正拍，<{RATIO_THRESHOLD}为斜拍）")
    print("=" * 80)

    summary_rows = []
    for s in strategies:
        st = agg[s.__name__]
        data = st["data"]

        if data:
            frontal = [(d, r, a) for d, r, a in data if r >= RATIO_THRESHOLD]
            tilted = [(d, r, a) for d, r, a in data if r < RATIO_THRESHOLD]

            all_deltas = [d for d, r, a in data]
            mean_all = sum(all_deltas) / len(all_deltas)
            max_all = max(all_deltas)
            mean_area = sum([a for d, r, a in data]) / len(data) * 100

            if frontal:
                frontal_deltas = [d for d, r, a in frontal]
                mean_frontal = sum(frontal_deltas) / len(frontal_deltas)
                max_frontal = max(frontal_deltas)
            else:
                mean_frontal = max_frontal = None

            if tilted:
                tilted_deltas = [d for d, r, a in tilted]
                mean_tilted = sum(tilted_deltas) / len(tilted_deltas)
                max_tilted = max(tilted_deltas)
            else:
                mean_tilted = max_tilted = None

            print(f"{s.__name__:<25}")
            print(f"  全部: N={len(data):>2} fail={st['fail']:>2} meanΔ={mean_all:.4f} maxΔ={max_all:.4f} 椭圆占比={mean_area:.0f}%")
            if frontal:
                print(f"  正拍: N={len(frontal):>2} meanΔ={mean_frontal:.4f} maxΔ={max_frontal:.4f}")
            if tilted:
                print(f"  斜拍: N={len(tilted):>2} meanΔ={mean_tilted:.4f} maxΔ={max_tilted:.4f}")

            summary_rows.append(
                f'<tr><td rowspan="3">{s.__name__}</td><td>全部</td><td>{len(data)}</td>'
                f'<td>{mean_all:.4f}</td><td>{max_all:.4f}</td><td>{mean_area:.0f}%</td></tr>'
                f'<tr><td>正拍≥{RATIO_THRESHOLD}</td><td>{len(frontal) if frontal else 0}</td>'
                f'<td>{fmt_num(mean_frontal)}</td>'
                f'<td>{fmt_num(max_frontal)}</td><td>—</td></tr>'
                f'<tr><td class="tilt">斜拍&lt;{RATIO_THRESHOLD}</td><td>{len(tilted) if tilted else 0}</td>'
                f'<td class="tilt">{fmt_num(mean_tilted)}</td>'
                f'<td class="tilt">{fmt_num(max_tilted)}</td><td>—</td></tr>'
            )
        else:
            print(f"{s.__name__:<25} 无有效数据 fail={st['fail']}")
            summary_rows.append(
                f'<tr><td>{s.__name__}</td><td>—</td><td>0</td><td>—</td><td>—</td><td>—</td></tr>'
            )

    summary_table = (
        '<table class="summary"><thead><tr>'
        '<th>策略</th><th>组别</th><th>N</th><th>平均Δ</th><th>最大Δ</th><th>椭圆占比</th>'
        '</tr></thead><tbody>'
        + "".join(summary_rows) +
        '</tbody></table>'
    )

    html_out = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>椭圆校正Mask策略对比 v2 (拆分正拍/斜拍)</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:#0a0c10;color:#e6e6e6;font:14px/1.5 sans-serif;padding:20px}}
h1{{font-size:22px;margin:0 0 8px;color:#7fd1ff}}
.meta{{color:#9aa4b2;font-size:13px;margin-bottom:20px}}
.summary{{border-collapse:collapse;margin-bottom:24px;background:#141821;border-radius:8px;overflow:hidden;width:100%}}
.summary th,.summary td{{padding:8px 14px;text-align:center;border:1px solid #232733;font-size:13px}}
.summary th{{background:#1a1f2a;color:#7fd1ff}}
.summary td:first-child{{text-align:left;color:#cbd5e1}}
.summary .tilt{{color:#ffcf6a;font-weight:600}}
.card{{background:#141821;border:1px solid #232733;border-radius:10px;padding:16px;margin-bottom:16px}}
.card h3{{font-size:15px;margin:0 0 12px;color:#cbd5e1}}
.grid{{display:flex;gap:8px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #232733}}
.comp{{display:flex;flex-direction:column;gap:10px}}
.row{{display:flex;gap:8px;align-items:center;background:#0e1117;padding:8px;border-radius:6px}}
.row.frontal{{border-left:3px solid #5ee08a}}
.row.tilted{{border-left:3px solid #ffcf6a}}
.lbl{{min-width:160px;font-size:12px;color:#9aa4b2}}
.lbl small{{display:block;font-size:10px;color:#7a8493}}
.lbl small.warn{{color:#ffcf6a}}
.ok{{color:#5ee08a}}
.fail{{color:#ff8b8b}}
figure{{margin:0;text-align:center}}
figure img{{height:{THUMB_H}px;border-radius:6px;border:1px solid #2a2f3a;display:block;background:#000}}
figure.miss .ph{{height:{THUMB_H}px;width:120px;display:flex;align-items:center;justify-content:center;color:#5a6472;border:1px dashed #2a2f3a;border-radius:6px;font-size:32px}}
figcaption{{font-size:10px;color:#7a8493;margin-top:4px}}
</style></head>
<body>
<h1>椭圆校正 Mask 生成策略对比实验 v2</h1>
<div class="meta">对比 HSV Baseline / HSV+Edge Fusion / Edge Direct 三种方案的mask质量、椭圆拟合、矫正效果和最终读数。<br>
<b>关键改进</b>: 按轴比≥{RATIO_THRESHOLD}拆分正拍/斜拍单独统计，绿边=正拍，黄边=斜拍。<br>
<b>重点看斜拍组</b>: 椭圆矫正的价值在斜拍上才体现，正拍的Δ应该小（不乱动），斜拍的Δ代表矫正幅度。</div>
{summary_table}
{''.join(cards)}
</body></html>"""

    out_path = os.path.join(PROJ, "test_results", "ellipse_mask_comparison_v2.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"\n报告已生成: {out_path}")
    print("卡片左侧绿边=正拍组，黄边=斜拍组")


if __name__ == "__main__":
    main()
