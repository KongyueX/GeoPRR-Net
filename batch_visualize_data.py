import argparse
import csv
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from utils.angleDetect.zeroShotMeter import meterZeroShot

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_INFER_CONFIG = {
    "scaleStart": 0.0,
    "scaleEnd": 1.6,
    "confidence": None,
    "use_origin_when_no_meter": False,
    "auto_zero": True,
    "auto_zero_threshold": 0.025,
    "start_end_distance_threshold": None,
    "start_end_position": "start_left_end_right",
    "snap_out_of_range_pointer": True,
    "correction_mode": "ransacFun",
    "stretch_x_ratio": 1.0,
    "stretch_y_ratio": 1.0,
    "reading_offset": 0.0,
    "default_start_angle": 45.0,
    "default_range_angle": 270.0,
    "validate_mask_line": True,
    "mask_center_threshold_ratio": 0.10,
    "visualization_mode": "debug",
}


def list_images(image_root: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        image_paths = []
        for root, _, files in os.walk(image_root):
            for file_name in files:
                if Path(file_name).suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(Path(root) / file_name)
        return sorted(image_paths)

    return sorted(
        path for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )



def read_image(image_path: Path):
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)



def save_image(image_path: Path, image) -> None:
    if image is None:
        return
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"图片编码失败: {image_path}")
    encoded.tofile(str(image_path))



def ensure_bgr(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image



def fit_into(image, width: int, height: int):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if image is None:
        cv2.putText(canvas, "N/A", (width // 2 - 25, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        return canvas

    image = ensure_bgr(image)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas



def draw_panel(title: str, image, width: int, height: int):
    header_h = 40
    body = fit_into(image, width, height - header_h)
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    panel[:header_h, :] = (235, 235, 235)
    panel[header_h:, :] = body
    cv2.putText(panel, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (190, 190, 190), 1)
    return panel



def make_summary_image(file_name: str, status_text: str, result_text: str, original, crop, result_image, mask_image):
    panel_w = 520
    panel_h = 380
    gap = 16
    top_h = 96
    width = panel_w * 2 + gap * 3
    height = top_h + panel_h * 2 + gap * 3
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)

    cv2.putText(canvas, file_name, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2)
    cv2.putText(canvas, status_text, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 60), 2)
    cv2.putText(canvas, result_text, (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 80, 20), 2)

    panels = [
        draw_panel("原图", original, panel_w, panel_h),
        draw_panel("表盘裁剪图", crop, panel_w, panel_h),
        draw_panel("结果图", result_image, panel_w, panel_h),
        draw_panel("分割掩码图", mask_image, panel_w, panel_h),
    ]

    positions = [
        (gap, top_h + gap),
        (gap * 2 + panel_w, top_h + gap),
        (gap, top_h + gap * 2 + panel_h),
        (gap * 2 + panel_w, top_h + gap * 2 + panel_h),
    ]

    for panel, (x, y) in zip(panels, positions):
        canvas[y:y + panel_h, x:x + panel_w] = panel

    return canvas



def build_zero_shot_model(device: str):
    base_dir = Path(__file__).resolve().parent / "utils" / "angleDetect"
    return meterZeroShot(
        str(base_dir / "pointerSeg" / "resultSeg" / "best.pt"),
        str(base_dir / "yoloDetection" / "result" / "yolo_findMeter.pt"),
        str(base_dir / "vitTranforms" / "result" / "best.pt"),
        str(base_dir / "yoloDetection" / "result" / "yolo_pointbest.pt"),
        device,
    )



def infer_one(model: meterZeroShot, image, config: dict):
    end_num, result_num, seg_pointer, result_image, origin_crop = model.Inference(
        image,
        config.get("scaleStart", 0.0),
        config.get("scaleEnd", 1.6),
        confidence=config.get("confidence"),
        use_origin_when_no_meter=config.get("use_origin_when_no_meter", False),
        start_end_distance_threshold=config.get("start_end_distance_threshold"),
        start_end_position=config.get("start_end_position", "start_left_end_right"),
        snap_out_of_range_pointer=config.get("snap_out_of_range_pointer", True),
        correction_mode=config.get("correction_mode", "ransacFun"),
        visualization_mode=config.get("visualization_mode", "debug"),
        stretch_x_ratio=config.get("stretch_x_ratio", 1.0),
        stretch_y_ratio=config.get("stretch_y_ratio", 1.0),
        reading_offset=config.get("reading_offset", 0.0),
        default_start_angle=config.get("default_start_angle", 45.0),
        default_range_angle=config.get("default_range_angle", 270.0),
        validate_mask_line=config.get("validate_mask_line", True),
        mask_center_threshold_ratio=config.get("mask_center_threshold_ratio", 0.10),
    )

    error = None
    if result_image is None:
        error = getattr(model, "last_error_message", None) or "推理失败"
    elif config.get("auto_zero", True):
        threshold = config.get("auto_zero_threshold", 0.025)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = 0.025
        if result_num < threshold:
            result_num = 0.0

    return end_num, result_num, seg_pointer, result_image, origin_crop, error



def safe_rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()



def html_img(path: Path, report_dir: Path) -> str:
    rel = os.path.relpath(path, report_dir).replace("\\", "/")
    return html.escape(rel)



def write_html_report(output_dir: Path, rows: list[dict]) -> None:
    report_path = output_dir / "report.html"
    parts = [
        "<!DOCTYPE html>",
        "<html lang='zh-CN'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>data 批量识别结果</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:24px;background:#f4f6f8;color:#1f2328;}",
        ".meta{margin-bottom:16px;color:#57606a;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:18px;}",
        ".card{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.06);}",
        ".card h3{margin:0 0 10px;font-size:16px;word-break:break-all;}",
        ".ok{color:#1a7f37;font-weight:700;}",
        ".bad{color:#cf222e;font-weight:700;}",
        ".thumb{width:100%;border:1px solid #d8dee4;border-radius:6px;background:#fff;}",
        ".kv{margin:8px 0 10px;line-height:1.6;}",
        ".paths{font-size:12px;color:#57606a;word-break:break-all;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>data 批量识别结果</h1>",
        f"<div class='meta'>共 {len(rows)} 张图像，报告目录：{html.escape(str(output_dir))}</div>",
        "<div class='grid'>",
    ]

    for row in rows:
        status_class = "ok" if row["status"] else "bad"
        result_text = "" if row["result"] is None else f"{row['result']:.6f}"
        summary_rel = html_img(output_dir / row["summary_image"], output_dir)
        parts.extend([
            "<div class='card'>",
            f"<h3>{html.escape(row['file_name'])}</h3>",
            f"<div class='kv'><span class='{status_class}'>{'成功' if row['status'] else '失败'}</span><br>",
            f"message: {html.escape(str(row['message']))}<br>",
            f"reading: {html.escape(result_text)}<br>",
            f"endNum: {html.escape('' if row['end_num'] is None else str(row['end_num']))}</div>",
            f"<a href='{summary_rel}' target='_blank'><img class='thumb' src='{summary_rel}' alt='summary'></a>",
            f"<div class='paths'>summary: {html.escape(row['summary_image'])}</div>",
            "</div>",
        ])

    parts.extend(["</div>", "</body>", "</html>"])
    report_path.write_text("\n".join(parts), encoding="utf-8")



def main() -> int:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="批量运行 data 目录中的表计图片，并导出可视化结果。")
    parser.add_argument("--image-dir", type=Path, default=repo_dir / "data", help="输入图片目录")
    parser.add_argument("--output-dir", type=Path, default=repo_dir / "data_batch_visualization", help="输出目录")
    parser.add_argument("--recursive", action="store_true", help="是否递归扫描图片")
    parser.add_argument("--device", default=None, help="运行设备，例如 cpu / cuda")
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir, recursive=args.recursive)
    if not images:
        raise FileNotFoundError(f"未找到图片: {image_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"开始加载模型，device={device}")
    model = build_zero_shot_model(device)
    logger.info(f"开始处理 {len(images)} 张图片")

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    rows = []

    for index, image_path in enumerate(images, start=1):
        logger.info(f"[{index}/{len(images)}] {image_path.name}")
        image = read_image(image_path)
        if image is None:
            row = {
                "file_name": image_path.name,
                "relative_path": safe_rel(image_path, image_dir),
                "status": False,
                "message": "读取图像失败",
                "result": None,
                "end_num": None,
                "summary_image": "",
                "original_image": "",
                "crop_image": "",
                "result_image": "",
                "mask_image": "",
            }
            rows.append(row)
            continue

        end_num, result_num, seg_pointer, result_image, origin_crop, error = infer_one(model, image, DEFAULT_INFER_CONFIG)
        item_dir = output_dir / image_path.stem
        item_dir.mkdir(parents=True, exist_ok=True)

        original_out = item_dir / "01_original.jpg"
        crop_out = item_dir / "02_crop.jpg"
        result_out = item_dir / "03_result.jpg"
        mask_out = item_dir / "04_mask.jpg"
        summary_out = item_dir / "00_summary.jpg"

        save_image(original_out, image)
        save_image(crop_out, origin_crop)
        save_image(result_out, result_image)
        save_image(mask_out, ensure_bgr(seg_pointer) if seg_pointer is not None else None)

        status = error is None and result_image is not None
        message = error or f"旋转角度为{end_num}，表盘读数是{result_num:.2f}"
        result_text = f"最终读数: {result_num:.4f}" if result_num is not None else "最终读数: N/A"
        summary_image = make_summary_image(
            image_path.name,
            f"状态: {'成功' if status else '失败'} | {message}",
            result_text,
            image,
            origin_crop,
            result_image,
            seg_pointer,
        )
        save_image(summary_out, summary_image)

        row = {
            "file_name": image_path.name,
            "relative_path": safe_rel(image_path, image_dir),
            "status": status,
            "message": message,
            "result": result_num,
            "end_num": end_num,
            "summary_image": safe_rel(summary_out, output_dir),
            "original_image": safe_rel(original_out, output_dir),
            "crop_image": safe_rel(crop_out, output_dir),
            "result_image": safe_rel(result_out, output_dir),
            "mask_image": safe_rel(mask_out, output_dir),
        }
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "relative_path",
                "status",
                "message",
                "result",
                "end_num",
                "summary_image",
                "original_image",
                "crop_image",
                "result_image",
                "mask_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html_report(output_dir, rows)

    success_count = sum(1 for row in rows if row["status"])
    logger.info(f"处理完成: success={success_count}, failed={len(rows) - success_count}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"HTML: {output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
