"""
批量对比新旧 U2Net 模型
为每张图片生成详细的可视化结果，并输出 HTML 报告
"""
import os
import cv2
import json
import numpy as np
from glob import glob
from loguru import logger
from utils.angleDetect.zeroShotMeter import meterZeroShot
from datetime import datetime


def create_side_by_side_comparison(old_data, new_data, img_name):
    """
    创建左右对比图：旧模型 vs 新模型
    """
    parts = []

    # 旧模型部分
    if old_data['corpImg'] is not None:
        old_corp = old_data['corpImg'].copy()
        old_mask = old_data['segPointer'].copy()

        if len(old_mask.shape) == 2:
            old_mask = cv2.cvtColor(old_mask, cv2.COLOR_GRAY2BGR)

        # 调整尺寸
        h, w = old_corp.shape[:2]
        if h > 400:
            scale = 400 / h
            old_corp = cv2.resize(old_corp, None, fx=scale, fy=scale)
            old_mask = cv2.resize(old_mask, None, fx=scale, fy=scale)

        # 标题
        title_h = 50
        title = np.ones((title_h, old_corp.shape[1], 3), dtype=np.uint8) * 240
        cv2.putText(title, "OLD MODEL (best.pt)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 信息
        info_h = 120
        info = np.ones((info_h, old_corp.shape[1], 3), dtype=np.uint8) * 250
        y_pos = 25
        cv2.putText(info, f"Reading: {old_data['resultNum']:.4f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        cv2.putText(info, f"EndNum: {old_data['endNum']}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        cv2.putText(info, f"Mask Distance: {old_data.get('mask_distance', 'N/A')}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        cv2.putText(info, f"Status: {'OK' if old_data['status'] else 'FAILED'}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0) if old_data['status'] else (0, 0, 255), 2)

        old_section = np.vstack([title, old_corp, old_mask, info])
        parts.append(old_section)

    # 新模型部分
    if new_data['corpImg'] is not None:
        new_corp = new_data['corpImg'].copy()
        new_mask = new_data['segPointer'].copy()

        if len(new_mask.shape) == 2:
            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)

        # 调整尺寸
        h, w = new_corp.shape[:2]
        if h > 400:
            scale = 400 / h
            new_corp = cv2.resize(new_corp, None, fx=scale, fy=scale)
            new_mask = cv2.resize(new_mask, None, fx=scale, fy=scale)

        # 标题
        title_h = 50
        title = np.ones((title_h, new_corp.shape[1], 3), dtype=np.uint8) * 240
        cv2.putText(title, "NEW MODEL (best_val_iou.pt)", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 信息
        info_h = 120
        info = np.ones((info_h, new_corp.shape[1], 3), dtype=np.uint8) * 250
        y_pos = 25
        cv2.putText(info, f"Reading: {new_data['resultNum']:.4f}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        cv2.putText(info, f"EndNum: {new_data['endNum']}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        cv2.putText(info, f"Mask Distance: {new_data.get('mask_distance', 'N/A')}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        cv2.putText(info, f"Status: {'OK' if new_data['status'] else 'FAILED'}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0) if new_data['status'] else (0, 0, 255), 2)

        new_section = np.vstack([title, new_corp, new_mask, info])
        parts.append(new_section)

    # 左右拼接
    if len(parts) == 2:
        max_h = max(p.shape[0] for p in parts)
        padded = []
        for p in parts:
            if p.shape[0] < max_h:
                pad = np.ones((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8) * 255
                p = np.vstack([p, pad])
            padded.append(p)

        comparison = np.hstack(padded)

        # 添加顶部标题
        header_h = 80
        header = np.ones((header_h, comparison.shape[1], 3), dtype=np.uint8) * 255
        cv2.putText(header, img_name, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # 计算差异
        if old_data['status'] and new_data['status']:
            diff = new_data['resultNum'] - old_data['resultNum']
            diff_pct = (diff / old_data['resultNum'] * 100) if old_data['resultNum'] != 0 else 0
            cv2.putText(header, f"Delta: {diff:.4f} ({diff_pct:+.2f}%)", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if abs(diff) > 0.05 else (0, 128, 0), 2)

        final = np.vstack([header, comparison])
        return final

    return None


def process_single_image(img_path, old_model, new_model, config, output_dir):
    """
    处理单张图片，生成完整的可视化结果
    """
    img_name = os.path.basename(img_path)
    img_base = os.path.splitext(img_name)[0]

    logger.info(f"处理: {img_name}")

    # 创建输出目录
    img_output_dir = os.path.join(output_dir, img_base)
    os.makedirs(img_output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(img_path)
    if image is None:
        logger.error(f"无法读取: {img_path}")
        return None

    # 保存原图
    cv2.imwrite(os.path.join(img_output_dir, "01_original.jpg"), image)

    # 旧模型推理
    old_endNum, old_resultNum, old_segPointer, old_corpImg, old_origin_crop = old_model.Inference(
        image,
        scaleStart=config['scaleStart'],
        scaleEnd=config['scaleEnd'],
        confidence=config.get('confidence'),
        correction_mode=config.get('correction_mode', 'ransacFun'),
    )

    # 获取旧模型的掩码距离
    old_mask_distance = getattr(old_model, '_last_mask_distance', None)

    old_data = {
        'status': old_corpImg is not None,
        'endNum': old_endNum,
        'resultNum': old_resultNum,
        'segPointer': old_segPointer,
        'corpImg': old_corpImg,
        'origin_crop': old_origin_crop,
        'mask_distance': f"{old_mask_distance:.2f}" if old_mask_distance else "N/A",
        'message': old_model.last_error_message if old_corpImg is None else f"旋转角度为{old_endNum}，表盘读数是{old_resultNum:.2f}"
    }

    # 新模型推理
    new_endNum, new_resultNum, new_segPointer, new_corpImg, new_origin_crop = new_model.Inference(
        image,
        scaleStart=config['scaleStart'],
        scaleEnd=config['scaleEnd'],
        confidence=config.get('confidence'),
        correction_mode=config.get('correction_mode', 'ransacFun'),
    )

    # 获取新模型的掩码距离
    new_mask_distance = getattr(new_model, '_last_mask_distance', None)

    new_data = {
        'status': new_corpImg is not None,
        'endNum': new_endNum,
        'resultNum': new_resultNum,
        'segPointer': new_segPointer,
        'corpImg': new_corpImg,
        'origin_crop': new_origin_crop,
        'mask_distance': f"{new_mask_distance:.2f}" if new_mask_distance else "N/A",
        'message': new_model.last_error_message if new_corpImg is None else f"旋转角度为{new_endNum}，表盘读数是{new_resultNum:.2f}"
    }

    # 保存裁剪图
    if old_origin_crop is not None:
        cv2.imwrite(os.path.join(img_output_dir, "02_crop.jpg"), old_origin_crop)

    # 保存旧模型结果
    if old_corpImg is not None:
        cv2.imwrite(os.path.join(img_output_dir, "03_old_result.jpg"), old_corpImg)
    if old_segPointer is not None:
        cv2.imwrite(os.path.join(img_output_dir, "04_old_mask.jpg"), old_segPointer)

    # 保存新模型结果
    if new_corpImg is not None:
        cv2.imwrite(os.path.join(img_output_dir, "05_new_result.jpg"), new_corpImg)
    if new_segPointer is not None:
        cv2.imwrite(os.path.join(img_output_dir, "06_new_mask.jpg"), new_segPointer)

    # 创建对比图
    comparison = create_side_by_side_comparison(old_data, new_data, img_name)
    if comparison is not None:
        cv2.imwrite(os.path.join(img_output_dir, "00_summary.jpg"), comparison)

    # 返回统计数据
    result = {
        'file_name': img_name,
        'old': {
            'status': old_data['status'],
            'message': old_data['message'],
            'result': old_resultNum,
            'end_num': old_endNum,
            'mask_distance': old_mask_distance
        },
        'new': {
            'status': new_data['status'],
            'message': new_data['message'],
            'result': new_resultNum,
            'end_num': new_endNum,
            'mask_distance': new_mask_distance
        },
        'status_changed': old_data['status'] != new_data['status'],
        'result_changed': False,
        'result_delta': None
    }

    if old_data['status'] and new_data['status']:
        result['result_changed'] = abs(old_resultNum - new_resultNum) > 0.001
        result['result_delta'] = new_resultNum - old_resultNum

    return result


def generate_html_report(results, output_dir, total_images):
    """
    生成 HTML 报告
    """
    html_template = """<!DOCTYPE html>
<html lang='zh-CN'>
<head>
<meta charset='utf-8'>
<title>U2Net 新旧模型对比报告</title>
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
.improved{{color:#0969da;font-weight:700;}}
.degraded{{color:#cf222e;font-weight:700;}}
.thumb{{width:100%;border:1px solid #d8dee4;border-radius:6px;background:#fff;}}
.kv{{margin:8px 0 10px;line-height:1.8;font-size:14px;}}
.paths{{font-size:12px;color:#57606a;word-break:break-all;margin-top:8px;}}
.delta{{background:#fff3cd;padding:8px;border-radius:4px;margin:8px 0;}}
</style>
</head>
<body>
<div class='header'>
<h1>U2Net 新旧模型对比报告</h1>
<div class='meta'>
生成时间: {timestamp}<br>
测试图片: {total_images} 张<br>
报告目录: {output_dir}
</div>
<div class='stats'>
<div class='stat-card'>
<h3>旧模型成功率</h3>
<div class='number'>{old_success}/{total_images}</div>
<div>{old_success_rate:.1f}%</div>
</div>
<div class='stat-card'>
<h3>新模型成功率</h3>
<div class='number'>{new_success}/{total_images}</div>
<div>{new_success_rate:.1f}%</div>
</div>
<div class='stat-card'>
<h3>状态改变</h3>
<div class='number'>{status_changed}</div>
<div>{improved} 改善 / {degraded} 退化</div>
</div>
<div class='stat-card'>
<h3>平均掩码距离改善</h3>
<div class='number'>{avg_mask_improvement:.1f}%</div>
</div>
</div>
</div>
<div class='grid'>
"""

    # 统计数据
    old_success = sum(1 for r in results if r['old']['status'])
    new_success = sum(1 for r in results if r['new']['status'])
    status_changed = sum(1 for r in results if r['status_changed'])

    improved = sum(1 for r in results if not r['old']['status'] and r['new']['status'])
    degraded = sum(1 for r in results if r['old']['status'] and not r['new']['status'])

    # 计算平均掩码距离改善
    mask_improvements = []
    for r in results:
        if r['old']['status'] and r['new']['status']:
            old_dist = r['old'].get('mask_distance')
            new_dist = r['new'].get('mask_distance')
            if old_dist and new_dist and old_dist > 0:
                improvement = (old_dist - new_dist) / old_dist * 100
                mask_improvements.append(improvement)

    avg_mask_improvement = sum(mask_improvements) / len(mask_improvements) if mask_improvements else 0

    html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        output_dir=os.path.abspath(output_dir),
        total_images=total_images,
        old_success=old_success,
        old_success_rate=old_success/total_images*100,
        new_success=new_success,
        new_success_rate=new_success/total_images*100,
        status_changed=status_changed,
        improved=improved,
        degraded=degraded,
        avg_mask_improvement=avg_mask_improvement
    )

    # 生成每张图片的卡片
    for r in results:
        img_base = os.path.splitext(r['file_name'])[0]

        # 状态标识
        old_status = "<span class='ok'>旧模型: 成功</span>" if r['old']['status'] else "<span class='bad'>旧模型: 失败</span>"
        new_status = "<span class='ok'>新模型: 成功</span>" if r['new']['status'] else "<span class='bad'>新模型: 失败</span>"

        status_change = ""
        if r['status_changed']:
            if not r['old']['status'] and r['new']['status']:
                status_change = "<span class='improved'>✓ 改善：失败 → 成功</span><br>"
            elif r['old']['status'] and not r['new']['status']:
                status_change = "<span class='degraded'>✗ 退化：成功 → 失败</span><br>"

        # 差异信息
        delta_info = ""
        if r['result_delta'] is not None:
            delta_pct = (r['result_delta'] / r['old']['result'] * 100) if r['old']['result'] != 0 else 0
            delta_info = f"<div class='delta'>读数差异: {r['result_delta']:+.4f} ({delta_pct:+.2f}%)</div>"

        # 掩码距离信息
        mask_info = ""
        if r['old']['status'] and r['new']['status']:
            old_mask = r['old'].get('mask_distance')
            new_mask = r['new'].get('mask_distance')
            if old_mask is not None and new_mask is not None:
                mask_improve = (old_mask - new_mask) / old_mask * 100 if old_mask > 0 else 0
                mask_info = f"掩码距离: {old_mask:.2f} → {new_mask:.2f} ({mask_improve:+.1f}%)<br>"

        card_html = f"""
<div class='card'>
<h3>{r['file_name']}</h3>
<div class='kv'>
{status_change}
{old_status}<br>
message: {r['old']['message']}<br>
reading: {r['old']['result']:.6f if r['old']['result'] is not None else 'N/A'}<br>
endNum: {r['old']['end_num'] if r['old']['end_num'] is not None else 'N/A'}<br>
<br>
{new_status}<br>
message: {r['new']['message']}<br>
reading: {r['new']['result']:.6f if r['new']['result'] is not None else 'N/A'}<br>
endNum: {r['new']['end_num'] if r['new']['end_num'] is not None else 'N/A'}<br>
{mask_info}
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

    with open(os.path.join(output_dir, "report.html"), 'w', encoding='utf-8') as f:
        f.write(html)


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
    logger.info("="*60)
    logger.info("加载旧模型 (best.pt)...")
    old_model = meterZeroShot(old_u2net_weights, yolo_weights, meterclip_weights,
                               point_weights, device)

    # 加载新模型
    logger.info("加载新模型 (best_val_iou.pt)...")
    new_model = meterZeroShot(new_u2net_weights, yolo_weights, meterclip_weights,
                               point_weights, device)
    logger.info("="*60)

    # 测试配置
    config = {
        'scaleStart': 0,
        'scaleEnd': 1.6,
        'confidence': None,
        'correction_mode': 'ransacFun',
    }

    # 获取所有测试图片
    image_dir = "data"
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

    logger.info(f"找到 {len(image_paths)} 张测试图片")

    # 输出目录
    output_dir = "u2net_comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    # 批量处理
    results = []
    for idx, img_path in enumerate(image_paths, 1):
        logger.info(f"\n[{idx}/{len(image_paths)}] 处理: {os.path.basename(img_path)}")
        result = process_single_image(img_path, old_model, new_model, config, output_dir)
        if result:
            results.append(result)

    # 保存 JSON 摘要
    summary = {
        'old_success': sum(1 for r in results if r['old']['status']),
        'old_failed': sum(1 for r in results if not r['old']['status']),
        'new_success': sum(1 for r in results if r['new']['status']),
        'new_failed': sum(1 for r in results if not r['new']['status']),
        'status_changed': [r for r in results if r['status_changed']],
        'improved': [r for r in results if not r['old']['status'] and r['new']['status']],
        'degraded': [r for r in results if r['old']['status'] and not r['new']['status']],
        'details': results
    }

    with open(os.path.join(output_dir, "comparison_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 生成 HTML 报告
    logger.info("\n生成 HTML 报告...")
    generate_html_report(results, output_dir, len(image_paths))

    # 输出统计
    logger.info("\n" + "="*60)
    logger.info("测试完成！")
    logger.info(f"总图片数: {len(image_paths)}")
    logger.info(f"旧模型成功: {summary['old_success']}/{len(image_paths)}")
    logger.info(f"新模型成功: {summary['new_success']}/{len(image_paths)}")
    logger.info(f"状态改变: {len(summary['status_changed'])}")
    logger.info(f"  - 改善（失败→成功）: {len(summary['improved'])}")
    logger.info(f"  - 退化（成功→失败）: {len(summary['degraded'])}")
    logger.info(f"\n报告已保存到: {os.path.abspath(output_dir)}")
    logger.info(f"HTML 报告: {os.path.join(output_dir, 'report.html')}")
    logger.info("="*60)
