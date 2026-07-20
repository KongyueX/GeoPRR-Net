# -*- coding: utf-8 -*-
"""
对指定目录下每张图跑推理，生成带"每一步中间过程"的 HTML 报告，方便逐步分析。

每张图展示：原图 -> 表盘crop -> 表盘面mask -> 椭圆拟合(含YOLO中心点) -> 矫正后 -> 指针分割 -> 标注(中心/起止/指针) -> 叠加分割 -> 叠加transformer射线
并对比 correction_mode = off / ellipse 两种下的最终读数。

v3 改进：展示YOLO检测的表盘中心点（蓝色标记），以及是否用于椭圆矫正。
v4 改进：增加步骤⑨显示transformer预测的指针射线（青色箭头），支持命令行指定数据目录。

运行:
  python report_data_html.py                              # 默认data目录
  python report_data_html.py data625                      # 指定data625目录
  python report_data_html.py data625 data_report_data625  # 指定输出文件名
"""
import os, sys, base64, html
import cv2
import numpy as np
import torch
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ)
from utils.angleDetect.zeroShotMeter import meterZeroShot

W = os.path.join(PROJ, "utils", "angleDetect")
WEIGHTS = dict(
    u2netWeights=os.path.join(W, "pointerSeg", "resultSeg", "best_val_iou626.pt"),  # 新微调权重
    yoloWeights=os.path.join(W, "yoloDetection", "result", "yolo_findMeter.pt"),
    meterclipWeights=os.path.join(W, "vitTranforms", "result", "best.pt"),
    pointWeights=os.path.join(W, "yoloDetection", "result", "yolo_pointbest.pt"),
)
SCALE_START, SCALE_END = 0, 1.6
THUMB_H = 200


def to_data_uri(img, h=THUMB_H):
    """cv2 BGR/灰度图 -> base64 jpg data uri；None 返回空。"""
    if img is None:
        return None
    im = img
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[0] != h:
        scale = h / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), h))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def overlay_seg_on_dial(base_bgr, seg_bgr, color=(0, 165, 255), alpha=0.45):
    """把指针分割掩码(半透明上色+描边)叠加到带标注的表盘图上，便于核对分割/点位准确性。"""
    if base_bgr is None:
        return seg_bgr
    out = base_bgr.copy()
    if seg_bgr is None:
        return out
    seg_gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY) if seg_bgr.ndim == 3 else seg_bgr
    if seg_gray.shape[:2] != out.shape[:2]:
        seg_gray = cv2.resize(seg_gray, (out.shape[1], out.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    m = seg_gray > 127
    if m.any():
        layer = np.zeros_like(out)
        layer[:] = color
        out[m] = (out[m] * (1 - alpha) + layer[m] * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def draw_yolo_center_on_ellipse(ellipse_overlay, yolo_center, hybrid_ellipse):
    """在椭圆拟合图上标记YOLO中心点（蓝色）和混合椭圆信息。"""
    if ellipse_overlay is None:
        return None
    out = ellipse_overlay.copy()

    if yolo_center is not None:
        # 蓝色大圆点标记YOLO中心
        cx, cy = int(yolo_center[0]), int(yolo_center[1])
        cv2.circle(out, (cx, cy), 7, (255, 0, 0), -1)  # 蓝色实心圆
        cv2.circle(out, (cx, cy), 8, (255, 255, 255), 1)  # 白色边框

        # 添加文字标签
        cv2.putText(out, "YOLO", (cx + 12, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(out, "YOLO", (cx + 12, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 如果有混合椭圆且与原椭圆不同，用虚线画出
    if hybrid_ellipse is not None:
        cv2.ellipse(out, hybrid_ellipse, (0, 255, 255), 1)  # 青色虚线

    return out


def draw_transformer_ray(base_img, pointer_ray):
    """
    在图像上绘制transformer模型预测的指针射线（青色箭头）。
    pointer_ray 为 (center_pt, end_pt) 或 None。返回叠加了射线的新图像。
    """
    if base_img is None:
        return base_img
    out = base_img.copy()
    if pointer_ray is not None:
        center_pt, end_pt = pointer_ray
        cv2.arrowedLine(
            out,
            tuple(map(int, center_pt)),
            tuple(map(int, end_pt)),
            (255, 255, 0),  # 青色箭头
            3,
            line_type=cv2.LINE_AA,
            tipLength=0.08,
        )
    return out


def fmt(v):
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def figure(uri, label):
    if uri is None:
        return (f'<figure class="step missing"><div class="ph">无</div>'
                f'<figcaption>{html.escape(label)}</figcaption></figure>')
    return (f'<figure class="step"><img src="{uri}" loading="lazy">'
            f'<figcaption>{html.escape(label)}</figcaption></figure>')


def main():
    # 解析命令行参数
    data_dir_name = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "data_report"

    data_dir = os.path.join(PROJ, data_dir_name)
    OUT_HTML = os.path.join(PROJ, "test_results", f"{output_name}.html")

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在 {data_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device= {device}")
    print(f"data_dir= {data_dir}")
    print(f"output= {OUT_HTML}")

    model = meterZeroShot(device=device, **WEIGHTS)

    imgs = sorted([f for f in os.listdir(data_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"images={len(imgs)}")

    cards = []
    n_meter_fail = 0
    n_applied = 0
    n_changed = 0
    n_yolo_used = 0  # 新增：YOLO中心使用次数
    sum_abs_delta = 0.0

    for i, name in enumerate(imgs):
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None:
            continue

        # off模式
        o_end, o_res, _, _, _ = model.Inference(img, SCALE_START, SCALE_END,
                                                correction_mode="off")

        # ellipse模式（带中间过程，含YOLO中心约束）
        # visualization_mode="standard"：annotated只含中心/起止参考，不含射线，
        # 射线在步骤⑨单独显式绘制，保证⑧/⑨干净分离。
        e_end, e_res, seg, annotated, crop = model.Inference(
            img, SCALE_START, SCALE_END,
            correction_mode="ellipse",
        )
        dbg = dict(model._correction_debug)
        pointer_ray = getattr(model, "_last_pointer_ray", None)

        applied = bool(dbg.get("applied"))
        yolo_center = dbg.get("yolo_center")
        hybrid_ellipse = dbg.get("hybrid_ellipse")
        yolo_used = (yolo_center is not None)

        if applied:
            n_applied += 1
        if yolo_used:
            n_yolo_used += 1

        meter_fail = (crop is None and o_res is None and e_res is None)
        if meter_fail:
            n_meter_fail += 1

        delta = None
        if o_res is not None and e_res is not None:
            delta = abs(float(e_res) - float(o_res))
            sum_abs_delta += delta
            if delta > 1e-3:
                n_changed += 1

        ratio_txt = ""
        el = dbg.get("ellipse")
        if el is not None:
            (_, _), (d1, d2), _ = el
            ratio_txt = f"轴比={min(d1, d2)/max(d1, d2):.3f}"

        # 在椭圆图上标记YOLO中心点
        ellipse_with_yolo = draw_yolo_center_on_ellipse(
            dbg.get("ellipse_overlay"), yolo_center, hybrid_ellipse)

        # 步骤⑧：分割+标注叠加（annotated + seg，无射线）
        overlay_8 = overlay_seg_on_dial(annotated, seg)

        # 步骤⑨：在⑧基础上增加transformer预测射线（青色箭头）
        overlay_9 = draw_transformer_ray(overlay_8, pointer_ray)

        steps = "".join([
            figure(to_data_uri(img), "① 原图"),
            figure(to_data_uri(crop), "② 表盘 crop"),
            figure(to_data_uri(dbg.get("face_mask")), "③ 表盘面 mask"),
            figure(to_data_uri(ellipse_with_yolo),
                   "④ 椭圆拟合 + YOLO中心(蓝点)"),
            figure(to_data_uri(dbg.get("rectified")), "⑤ 矫正后"),
            figure(to_data_uri(seg), "⑥ 指针分割"),
            figure(to_data_uri(annotated), "⑦ 标注(中心/起止点)"),
            figure(to_data_uri(overlay_8),
                   "⑧ 分割+标注叠加(核对分割准确性)"),
            figure(to_data_uri(overlay_9),
                   "⑨ 完整结果(⑧ + transformer射线)"),
        ])

        # Badge显示：椭圆矫正状态 + YOLO中心使用状态
        badge_parts = []
        if applied:
            badge_parts.append('<span class="b applied">椭圆矫正已应用</span>')
        elif meter_fail:
            badge_parts.append('<span class="b fail">未检测到表盘</span>')
        else:
            badge_parts.append('<span class="b skip">未矫正(够圆/回退)</span>')

        if yolo_used:
            badge_parts.append('<span class="b yolo">YOLO中心已用</span>')
        elif not meter_fail:
            badge_parts.append('<span class="b yolo-no">YOLO中心未检测</span>')

        badge = ''.join(badge_parts)

        delta_cls = "diff" if (delta is not None and delta > 0.02) else ""
        reading = (
            f'<div class="readings">'
            f'<div class="rd"><span>off</span><b>{fmt(o_res)}</b>'
            f'<small>endNum {fmt(o_end)}</small></div>'
            f'<div class="rd ell"><span>ellipse + YOLO</span><b>{fmt(e_res)}</b>'
            f'<small>endNum {fmt(e_end)}</small></div>'
            f'<div class="rd {delta_cls}"><span>|Δ读数|</span><b>{fmt(delta)}</b>'
            f'<small>{ratio_txt}</small></div>'
            f'</div>'
        )

        cards.append(
            f'<section class="card">'
            f'<header><h3>#{i+1} {html.escape(name)}</h3><div class="badges">{badge}</div></header>'
            f'{reading}'
            f'<div class="strip">{steps}</div>'
            f'</section>'
        )
        yolo_mark = "✓" if yolo_used else "✗"
        print(f"[{i+1}/{len(imgs)}] {name} off={fmt(o_res)} ell={fmt(e_res)} "
              f"applied={applied} YOLO={yolo_mark} {ratio_txt}")

    yolo_rate = (n_yolo_used / len(imgs) * 100) if len(imgs) > 0 else 0
    summary = (
        f'<div class="summary">'
        f'<div><b>{len(imgs)}</b><span>总图数</span></div>'
        f'<div><b>{n_applied}</b><span>椭圆矫正应用</span></div>'
        f'<div><b>{n_yolo_used}</b><span>YOLO中心使用</span></div>'
        f'<div class="rate"><b>{yolo_rate:.1f}%</b><span>YOLO命中率</span></div>'
        f'<div><b>{n_changed}</b><span>读数变化(>0.001)</span></div>'
        f'<div><b>{n_meter_fail}</b><span>未检测到表盘</span></div>'
        f'<div><b>{(sum_abs_delta/len(imgs)):.4f}</b><span>平均|Δ读数|</span></div>'
        f'</div>'
    )

    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>表计读数中间过程报告 (v4 - {data_dir_name})</title>
<style>
:root{{color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:#0f1115;color:#e6e6e6;font:14px/1.5 -apple-system,Segoe UI,Roboto,Microsoft YaHei,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}}
.head{{padding:18px 22px;border-bottom:1px solid #232733;position:sticky;top:0;background:#0f1115;z-index:5}}
.meta{{color:#9aa4b2;font-size:12px}}
.summary{{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}}
.summary div{{background:#171a21;border:1px solid #232733;border-radius:10px;padding:8px 14px;text-align:center;min-width:96px}}
.summary b{{display:block;font-size:20px;color:#7fd1ff}}
.summary span{{font-size:11px;color:#9aa4b2}}
.summary .rate b{{color:#7dd3fc}}
.wrap{{padding:18px 22px;display:flex;flex-direction:column;gap:16px}}
.card{{background:#141821;border:1px solid #232733;border-radius:12px;padding:12px 14px}}
.card header{{display:flex;align-items:center;gap:10px;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap}}
.card h3{{font-size:14px;margin:0;color:#cbd5e1;font-weight:600}}
.badges{{display:flex;gap:6px;flex-wrap:wrap}}
.b{{font-size:11px;padding:3px 9px;border-radius:20px;white-space:nowrap}}
.b.applied{{background:#16361f;color:#5ee08a;border:1px solid #1f5230}}
.b.skip{{background:#2a2f17;color:#d9d36a;border:1px solid #4d521f}}
.b.fail{{background:#3a1a1a;color:#ff8b8b;border:1px solid #5a2626}}
.b.yolo{{background:#1a2a3a;color:#7dd3fc;border:1px solid#2a4a6a}}
.b.yolo-no{{background:#2a2a2a;color:#8a8a8a;border:1px solid#3a3a3a}}
.readings{{display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap}}
.rd{{background:#0e1117;border:1px solid #232733;border-radius:9px;padding:6px 12px;min-width:120px}}
.rd span{{font-size:11px;color:#9aa4b2;display:block}}
.rd b{{font-size:18px;color:#e6e6e6}}
.rd small{{display:block;font-size:10px;color:#7a8493}}
.rd.ell b{{color:#7fd1ff}}
.rd.diff{{border-color:#7a5a1f;background:#241c0d}}
.rd.diff b{{color:#ffcf6a}}
.strip{{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}}
.step{{margin:0;flex:0 0 auto;text-align:center}}
.step img{{height:{THUMB_H}px;border-radius:8px;border:1px solid #2a2f3a;display:block;background:#000;cursor:zoom-in}}
.step figcaption{{font-size:11px;color:#9aa4b2;margin-top:4px;max-width:240px}}
.step.missing .ph{{height:{THUMB_H}px;width:160px;display:flex;align-items:center;justify-content:center;color:#5a6472;border:1px dashed #2a2f3a;border-radius:8px}}
img.zoom{{position:fixed;inset:0;margin:auto;max-width:96vw;max-height:96vh;height:auto;z-index:50;box-shadow:0 0 0 100vmax rgba(0,0,0,.85);cursor:zoom-out}}
</style></head>
<body>
<div class="head">
  <h1>表计读数 · 每一步中间过程报告 (v4 - {data_dir_name})</h1>
  <div class="meta">量程 {SCALE_START}~{SCALE_END} · correction_mode 对比 off / ellipse(表盘面椭圆自动矫正 + YOLO中心点约束) · 蓝色点=YOLO检测中心 · 青色箭头=transformer预测射线 · 步骤图可点击放大</div>
  {summary}
</div>
<div class="wrap">
{''.join(cards)}
</div>
<script>
document.addEventListener('click',function(e){{
  var t=e.target;
  if(t.tagName==='IMG'&&t.closest('.step')){{
    if(t.classList.contains('zoom')){{t.classList.remove('zoom');}}
    else{{document.querySelectorAll('img.zoom').forEach(function(z){{z.classList.remove('zoom')}});t.classList.add('zoom');}}
  }} else {{document.querySelectorAll('img.zoom').forEach(function(z){{z.classList.remove('zoom')}});}}
}});
</script>
</body></html>"""

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"\n报告已生成: {OUT_HTML}")
    print(f"总图 {len(imgs)} | 椭圆应用 {n_applied} | YOLO中心 {n_yolo_used} ({yolo_rate:.1f}%) | "
          f"读数变化 {n_changed} | 未检测到表盘 {n_meter_fail} | 平均|Δ| {(sum_abs_delta/len(imgs)):.4f}")


if __name__ == "__main__":
    main()
