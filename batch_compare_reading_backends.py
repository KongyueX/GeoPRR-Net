import json
import os
from datetime import datetime
from glob import glob

import cv2
import numpy as np
from loguru import logger

from utils.angleDetect.zeroShotMeter import meterZeroShot


def create_side_by_side_comparison(transformer_data, geometry_data, img_name):
    parts = []

    def build_section(title_text, title_color, data):
        if data['corpImg'] is None:
            return None

        corp = data['corpImg'].copy()
        mask = data['segPointer'].copy() if data['segPointer'] is not None else None
        if mask is None:
            mask = np.zeros_like(corp)
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        h, _ = corp.shape[:2]
        if h > 400:
            scale = 400 / h
            corp = cv2.resize(corp, None, fx=scale, fy=scale)
            mask = cv2.resize(mask, None, fx=scale, fy=scale)

        title_h = 50
        title = np.ones((title_h, corp.shape[1], 3), dtype=np.uint8) * 240
        cv2.putText(title, title_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2)

        info_h = 140
        info = np.ones((info_h, corp.shape[1], 3), dtype=np.uint8) * 250
        y_pos = 25
        reading = data['resultNum']
        reading_text = 'N/A' if reading is None else f"{reading:.4f}"
        cv2.putText(info, f"Reading: {reading_text}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        cv2.putText(info, f"EndNum: {data.get('endNum', 'N/A')}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        cv2.putText(info, f"Backend: {data.get('backend', 'N/A')}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        y_pos += 30
        status_text = 'OK' if data['status'] else 'FAILED'
        status_color = (0, 128, 0) if data['status'] else (0, 0, 255)
        cv2.putText(info, f"Status: {status_text}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y_pos += 30
        message = (data.get('message') or '')[:40]
        cv2.putText(info, message, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        return np.vstack([title, corp, mask, info])

    transformer_section = build_section('TRANSFORMER', (0, 180, 0), transformer_data)
    geometry_section = build_section('GEOMETRY', (200, 90, 0), geometry_data)

    if transformer_section is not None:
        parts.append(transformer_section)
    if geometry_section is not None:
        parts.append(geometry_section)

    if len(parts) != 2:
        return None

    max_h = max(p.shape[0] for p in parts)
    padded = []
    for p in parts:
        if p.shape[0] < max_h:
            pad = np.ones((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8) * 255
            p = np.vstack([p, pad])
        padded.append(p)

    comparison = np.hstack(padded)
    header_h = 80
    header = np.ones((header_h, comparison.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(header, img_name, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    if transformer_data['status'] and geometry_data['status']:
        diff = geometry_data['resultNum'] - transformer_data['resultNum']
        diff_pct = (diff / transformer_data['resultNum'] * 100) if transformer_data['resultNum'] not in (None, 0) else 0
        color = (255, 0, 0) if abs(diff) > 0.05 else (0, 128, 0)
        cv2.putText(header, f"Delta: {diff:+.4f} ({diff_pct:+.2f}%)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return np.vstack([header, comparison])


def process_single_image(img_path, model, config, output_dir):
    img_name = os.path.basename(img_path)
    img_base = os.path.splitext(img_name)[0]
    logger.info(f"处理: {img_name}")

    img_output_dir = os.path.join(output_dir, img_base)
    os.makedirs(img_output_dir, exist_ok=True)

    image = cv2.imread(img_path)
    if image is None:
        logger.error(f"无法读取: {img_path}")
        return None

    cv2.imwrite(os.path.join(img_output_dir, '01_original.jpg'), image)

    shared_kwargs = dict(
        scaleStart=config['scaleStart'],
        scaleEnd=config['scaleEnd'],
        confidence=config.get('confidence'),
        correction_mode=config.get('correction_mode', 'ransacFun'),
        use_origin_when_no_meter=config.get('use_origin_when_no_meter', False),
        start_end_distance_threshold=config.get('start_end_distance_threshold'),
        start_end_position=config.get('start_end_position', 'start_left_end_right'),
        snap_out_of_range_pointer=config.get('snap_out_of_range_pointer', True),
        stretch_x_ratio=config.get('stretch_x_ratio', 1.0),
        stretch_y_ratio=config.get('stretch_y_ratio', 1.0),
        reading_offset=config.get('reading_offset', 0.0),
        default_start_angle=config.get('default_start_angle', 45.0),
        default_range_angle=config.get('default_range_angle', 270.0),
        validate_mask_line=config.get('validate_mask_line', True),
        mask_center_threshold_ratio=config.get('mask_center_threshold_ratio', 0.10),
        visualization_mode='debug',
    )

    transformer_endNum, transformer_resultNum, transformer_segPointer, transformer_corpImg, transformer_origin_crop = model.Inference(
        image,
        reading_backend='transformer',
        **shared_kwargs,
    )
    transformer_details = getattr(model, '_last_reading_details', {}) or {}
    transformer_data = {
        'status': transformer_corpImg is not None,
        'backend': 'transformer',
        'endNum': transformer_endNum,
        'resultNum': transformer_resultNum,
        'segPointer': transformer_segPointer,
        'corpImg': transformer_corpImg,
        'origin_crop': transformer_origin_crop,
        'message': model.last_error_message if transformer_corpImg is None else transformer_details.get('selected', {}).get('message', f"旋转角度为{transformer_endNum}，表盘读数是{transformer_resultNum:.2f}"),
        'details': transformer_details,
    }

    geometry_endNum, geometry_resultNum, geometry_segPointer, geometry_corpImg, geometry_origin_crop = model.Inference(
        image,
        reading_backend='geometry',
        **shared_kwargs,
    )
    geometry_details = getattr(model, '_last_reading_details', {}) or {}
    geometry_data = {
        'status': geometry_corpImg is not None,
        'backend': 'geometry',
        'endNum': geometry_endNum,
        'resultNum': geometry_resultNum,
        'segPointer': geometry_segPointer,
        'corpImg': geometry_corpImg,
        'origin_crop': geometry_origin_crop,
        'message': model.last_error_message if geometry_corpImg is None else geometry_details.get('selected', {}).get('message', f"旋转角度为{geometry_endNum}，表盘读数是{geometry_resultNum:.2f}"),
        'details': geometry_details,
    }

    if transformer_origin_crop is not None:
        cv2.imwrite(os.path.join(img_output_dir, '02_crop.jpg'), transformer_origin_crop)
    elif geometry_origin_crop is not None:
        cv2.imwrite(os.path.join(img_output_dir, '02_crop.jpg'), geometry_origin_crop)

    if transformer_corpImg is not None:
        cv2.imwrite(os.path.join(img_output_dir, '03_transformer_result.jpg'), transformer_corpImg)
    if transformer_segPointer is not None:
        cv2.imwrite(os.path.join(img_output_dir, '04_transformer_mask.jpg'), transformer_segPointer)
    if geometry_corpImg is not None:
        cv2.imwrite(os.path.join(img_output_dir, '05_geometry_result.jpg'), geometry_corpImg)
    if geometry_segPointer is not None:
        cv2.imwrite(os.path.join(img_output_dir, '06_geometry_mask.jpg'), geometry_segPointer)

    comparison = create_side_by_side_comparison(transformer_data, geometry_data, img_name)
    if comparison is not None:
        cv2.imwrite(os.path.join(img_output_dir, '00_summary.jpg'), comparison)

    result = {
        'file_name': img_name,
        'transformer': {
            'status': transformer_data['status'],
            'message': transformer_data['message'],
            'result': transformer_resultNum,
            'end_num': transformer_endNum,
        },
        'geometry': {
            'status': geometry_data['status'],
            'message': geometry_data['message'],
            'result': geometry_resultNum,
            'end_num': geometry_endNum,
        },
        'status_changed': transformer_data['status'] != geometry_data['status'],
        'result_changed': False,
        'result_delta': None,
    }
    if transformer_data['status'] and geometry_data['status']:
        result['result_changed'] = abs(transformer_resultNum - geometry_resultNum) > 0.001
        result['result_delta'] = geometry_resultNum - transformer_resultNum

    return result


def generate_html_report(results, output_dir, total_images):
    html_template = """<!DOCTYPE html>
<html lang='zh-CN'>
<head>
<meta charset='utf-8'>
<title>Transformer vs Geometry 对比报告</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:24px;background:#f4f6f8;color:#1f2328;}}
.header{{background:#fff;padding:20px;border-radius:10px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.06);}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0;}}
.stat-card{{background:#f0f6fc;border:1px solid #0969da;border-radius:8px;padding:16px;text-align:center;}}
.stat-card h3{{margin:0 0 8px;font-size:14px;color:#0969da;}}
.stat-card .number{{font-size:32px;font-weight:700;color:#0969da;}}
.meta{{margin-bottom:16px;color:#57606a;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:18px;}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.06);}}
.card h3{{margin:0 0 10px;font-size:16px;word-break:break-all;}}
.ok{{color:#1a7f37;font-weight:700;}}
.bad{{color:#cf222e;font-weight:700;}}
.changed{{color:#0969da;font-weight:700;}}
.thumb{{width:100%;border:1px solid #d8dee4;border-radius:6px;background:#fff;}}
.kv{{margin:8px 0 10px;line-height:1.8;font-size:14px;}}
.paths{{font-size:12px;color:#57606a;word-break:break-all;margin-top:8px;}}
.delta{{background:#fff3cd;padding:8px;border-radius:4px;margin:8px 0;}}
</style>
</head>
<body>
<div class='header'>
<h1>Transformer vs Geometry 对比报告</h1>
<div class='meta'>
生成时间: {timestamp}<br>
测试图片: {total_images} 张<br>
报告目录: {output_dir}
</div>
<div class='stats'>
<div class='stat-card'>
<h3>Transformer 成功率</h3>
<div class='number'>{transformer_success}/{total_images}</div>
<div>{transformer_success_rate:.1f}%</div>
</div>
<div class='stat-card'>
<h3>Geometry 成功率</h3>
<div class='number'>{geometry_success}/{total_images}</div>
<div>{geometry_success_rate:.1f}%</div>
</div>
<div class='stat-card'>
<h3>状态变化</h3>
<div class='number'>{status_changed}</div>
<div>{improved} 改善 / {degraded} 退化</div>
</div>
<div class='stat-card'>
<h3>平均差异</h3>
<div class='number'>{avg_delta:.4f}</div>
</div>
</div>
</div>
<div class='grid'>
"""

    transformer_success = sum(1 for r in results if r['transformer']['status'])
    geometry_success = sum(1 for r in results if r['geometry']['status'])
    status_changed = sum(1 for r in results if r['status_changed'])
    improved = sum(1 for r in results if not r['transformer']['status'] and r['geometry']['status'])
    degraded = sum(1 for r in results if r['transformer']['status'] and not r['geometry']['status'])
    deltas = [abs(r['result_delta']) for r in results if r['result_delta'] is not None]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

    html = html_template.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        output_dir=os.path.abspath(output_dir),
        total_images=total_images,
        transformer_success=transformer_success,
        transformer_success_rate=transformer_success / total_images * 100,
        geometry_success=geometry_success,
        geometry_success_rate=geometry_success / total_images * 100,
        status_changed=status_changed,
        improved=improved,
        degraded=degraded,
        avg_delta=avg_delta,
    )

    for r in results:
        img_base = os.path.splitext(r['file_name'])[0]
        transformer_status = "<span class='ok'>Transformer: 成功</span>" if r['transformer']['status'] else "<span class='bad'>Transformer: 失败</span>"
        geometry_status = "<span class='ok'>Geometry: 成功</span>" if r['geometry']['status'] else "<span class='bad'>Geometry: 失败</span>"
        status_change = ""
        if r['status_changed']:
            status_change = "<span class='changed'>状态发生变化</span><br>"
        delta_info = ""
        if r['result_delta'] is not None:
            delta_pct = (r['result_delta'] / r['transformer']['result'] * 100) if r['transformer']['result'] not in (None, 0) else 0
            delta_info = f"<div class='delta'>读数差异: {r['result_delta']:+.4f} ({delta_pct:+.2f}%)</div>"

        card_html = f"""
<div class='card'>
<h3>{r['file_name']}</h3>
<div class='kv'>
{status_change}
{transformer_status}<br>
message: {r['transformer']['message']}<br>
reading: {r['transformer']['result'] if r['transformer']['result'] is not None else 'N/A'}<br>
endNum: {r['transformer']['end_num'] if r['transformer']['end_num'] is not None else 'N/A'}<br>
<br>
{geometry_status}<br>
message: {r['geometry']['message']}<br>
reading: {r['geometry']['result'] if r['geometry']['result'] is not None else 'N/A'}<br>
endNum: {r['geometry']['end_num'] if r['geometry']['end_num'] is not None else 'N/A'}<br>
</div>
{delta_info}
<a href='{img_base}/00_summary.jpg' target='_blank'><img class='thumb' src='{img_base}/00_summary.jpg' alt='summary'></a>
<div class='paths'>summary: {img_base}/00_summary.jpg</div>
</div>
"""
        html += card_html

    html += """
</div>
</body>
</html>
"""

    with open(os.path.join(output_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == '__main__':
    filePath = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(filePath, 'utils')

    u2net_weights = os.path.join(weights_path, 'angleDetect/pointerSeg/resultSeg/best.pt')
    yolo_weights = os.path.join(weights_path, 'angleDetect/yoloDetection/result/yolo_findMeter.pt')
    meterclip_weights = os.path.join(weights_path, 'angleDetect/vitTranforms/result/best.pt')
    point_weights = os.path.join(weights_path, 'angleDetect/yoloDetection/result/yolo_pointbest.pt')

    device = 'cpu'
    logger.info('加载模型...')
    model = meterZeroShot(u2net_weights, yolo_weights, meterclip_weights, point_weights, device)

    config = {
        'scaleStart': 0,
        'scaleEnd': 1.6,
        'confidence': None,
        'correction_mode': 'ransacFun',
        'validate_mask_line': True,
        'mask_center_threshold_ratio': 0.10,
    }

    image_dir = 'data'
    image_paths = sorted(
        p for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        for p in glob(os.path.join(image_dir, ext))
    )
    logger.info(f'找到 {len(image_paths)} 张测试图片')

    output_dir = 'reading_backend_comparison_results'
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for idx, img_path in enumerate(image_paths, 1):
        logger.info(f'[{idx}/{len(image_paths)}] 处理: {os.path.basename(img_path)}')
        result = process_single_image(img_path, model, config, output_dir)
        if result:
            results.append(result)

    summary = {
        'transformer_success': sum(1 for r in results if r['transformer']['status']),
        'transformer_failed': sum(1 for r in results if not r['transformer']['status']),
        'geometry_success': sum(1 for r in results if r['geometry']['status']),
        'geometry_failed': sum(1 for r in results if not r['geometry']['status']),
        'status_changed': [r for r in results if r['status_changed']],
        'details': results,
    }
    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info('生成 HTML 报告...')
    generate_html_report(results, output_dir, len(image_paths))
    logger.info(f'报告已保存到: {os.path.abspath(output_dir)}')
