import argparse
import hashlib
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logger = logging.getLogger(__name__)

filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CORRECTION_MODES = [
    ("off", "关闭"),
    ("ransacFun", "ransacFun"),
    ("ransacFunbackup", "ransacFunbackup"),
    ("square", "square"),
    ("stretch", "stretch"),
]

DEFAULT_INFER_CONFIG = {
    "scaleStart": 0.0,
    "scaleEnd": 1.6,
    "confidence": "",
    "use_origin_when_no_meter": False,
    "auto_zero": True,
    "auto_zero_threshold": 0.025,
    "start_end_distance_threshold": "",
    "start_end_position": "start_left_end_right",
    "snap_out_of_range_pointer": True,
    "stretch_x_ratio": 1.0,
    "stretch_y_ratio": 1.0,
    "reading_offset": 0.0,
    "default_start_angle": 45.0,
    "default_range_angle": 270.0,
    "validate_mask_line": True,
    "mask_center_threshold_ratio": 0.10,
}
OPTIONAL_FLOAT_KEYS = {"confidence", "start_end_distance_threshold"}


def list_images(image_root, recursive=False):
    if recursive:
        image_paths = []
        for root, _, files in os.walk(image_root):
            for file_name in files:
                if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, file_name))
    else:
        image_paths = [
            os.path.join(image_root, file_name)
            for file_name in os.listdir(image_root)
            if os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS
        ]
    return sorted(image_paths)


def read_image(image_path):
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def ensure_bgr(image):
    if image is None:
        return None
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def cv_image_to_photo(image, max_width, max_height):
    image = ensure_bgr(image)
    if image is None:
        pil_image = Image.new("RGB", (max_width, max_height), (32, 32, 32))
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((max_width, max_height), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_image)


def build_zero_shot_model(device):
    from zeroShotMeter import meterZeroShot

    u2netWeights = os.path.join(filePath, "pointerSeg", "resultSeg", "best.pt")
    yoloWeights = os.path.join(filePath, "yoloDetection", "result", "yolo_findMeter.pt")
    meterclipWeights = os.path.join(filePath, "vitTranforms", "result", "best.pt")
    pointWeights = os.path.join(filePath, "yoloDetection", "result", "yolo_pointbest.pt")

    return meterZeroShot(
        u2netWeights,
        yoloWeights,
        meterclipWeights,
        pointWeights,
        device,
    )


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def clean_config_for_infer(config):
    cleaned = {}
    for key, value in config.items():
        if value == "":
            continue
        cleaned[key] = value
    return cleaned


class QuickBrowseApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.image_root = os.path.abspath(args.image_root)
        self.config_path = os.path.abspath(args.config or os.path.join(self.image_root, "quick_browse_inferconfig.json"))
        self.image_paths = list_images(self.image_root, args.recursive)
        if not self.image_paths:
            raise FileNotFoundError(f"image_root 下未找到图像: {self.image_root}")

        self.index = 0
        self.cache = {}
        self.photo_refs = []
        self.global_config = DEFAULT_INFER_CONFIG.copy()
        self.per_index_config = {}
        self.saved_image_order = []
        self.model = None

        self.root.title("Pointer Meter Quick Browse")
        self.root.geometry("1500x980")
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("a", lambda event: self.prev_image())
        self.root.bind("d", lambda event: self.next_image())

        self._build_ui()
        self.load_config(silent=True)
        self._load_config_to_vars(self._get_current_config())
        self.root.after(100, self.render_current_image)

    def _build_ui(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        self.main_frame = ttk.Frame(outer)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(outer, padding=8)
        controls.pack(side=tk.RIGHT, fill=tk.Y)

        self.header_var = tk.StringVar()
        ttk.Label(self.main_frame, textvariable=self.header_var, font=("Microsoft YaHei", 12, "bold")).pack(anchor="w", padx=8, pady=6)

        canvas = tk.Canvas(self.main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.rows_container = ttk.Frame(canvas)
        self.rows_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.rows_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(controls, text="Infer Config", font=("Microsoft YaHei", 12, "bold")).pack(anchor="w", pady=(0, 8))
        self.vars = {}
        self._add_entry(controls, "scaleStart")
        self._add_entry(controls, "scaleEnd")
        self._add_entry(controls, "confidence")
        self._add_entry(controls, "auto_zero_threshold")
        self._add_entry(controls, "start_end_distance_threshold")
        self._add_entry(controls, "start_end_position")
        self._add_entry(controls, "stretch_x_ratio")
        self._add_entry(controls, "stretch_y_ratio")
        self._add_entry(controls, "reading_offset")
        self._add_entry(controls, "default_start_angle")
        self._add_entry(controls, "default_range_angle")
        self._add_entry(controls, "mask_center_threshold_ratio")
        self._add_check(controls, "use_origin_when_no_meter")
        self._add_check(controls, "auto_zero")
        self._add_check(controls, "snap_out_of_range_pointer")
        self._add_check(controls, "validate_mask_line")

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Button(controls, text="Apply / Rerun", command=self.apply_and_rerun).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="上一张 A/←", command=self.prev_image).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="下一张 D/→", command=self.next_image).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="保存配置", command=self.save_config).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="读取配置", command=lambda: self.load_config(silent=False)).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="另存配置", command=self.save_config_as).pack(fill=tk.X, pady=2)
        ttk.Button(controls, text="复制当前配置到全部", command=self.copy_current_to_all).pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.status_var, wraplength=280, foreground="#555").pack(anchor="w", pady=(10, 0))

    def _add_entry(self, parent, key):
        ttk.Label(parent, text=key).pack(anchor="w")
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=28).pack(fill=tk.X, pady=(0, 4))

    def _add_check(self, parent, key):
        var = tk.BooleanVar()
        self.vars[key] = var
        ttk.Checkbutton(parent, text=key, variable=var).pack(anchor="w", pady=2)

    def _get_image_order(self):
        return [
            {
                "index": idx,
                "relative_path": os.path.relpath(path, self.image_root).replace("\\", "/"),
            }
            for idx, path in enumerate(self.image_paths)
        ]

    def _get_current_config(self):
        config = self.global_config.copy()
        config.update(self.per_index_config.get(str(self.index), {}))
        return config

    def _load_config_to_vars(self, config):
        for key, default_value in DEFAULT_INFER_CONFIG.items():
            value = config.get(key, default_value)
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                var.set(parse_bool(value))
            else:
                var.set("" if value is None else str(value))

    def _read_vars_config(self):
        config = {}
        for key, default_value in DEFAULT_INFER_CONFIG.items():
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                config[key] = bool(var.get())
                continue

            raw = var.get().strip()
            if raw == "":
                config[key] = ""
            elif isinstance(default_value, float) or key in OPTIONAL_FLOAT_KEYS:
                try:
                    config[key] = float(raw)
                except ValueError:
                    config[key] = default_value
            else:
                config[key] = raw
        return config

    def _config_hash(self, config, correction_mode):
        payload = config.copy()
        payload["correction_mode"] = correction_mode
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(encoded.encode("utf-8")).hexdigest()

    def _ensure_model(self):
        if self.model is not None:
            return
        import torch

        device = self.args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.status_var.set(f"加载模型 device={device}")
        self.root.update_idletasks()
        self.model = build_zero_shot_model(device)

    def infer_one(self, image_path, config, correction_mode):
        self._ensure_model()
        image = read_image(image_path)
        if image is None:
            return None, None, None, None, None, "读取图像失败", None

        infer_config = clean_config_for_infer(config)
        try:
            endNum, resultNum, segPointer, corpImg, origin_crop = self.model.Inference(
                image,
                infer_config.get("scaleStart", 0.0),
                infer_config.get("scaleEnd", 1.6),
                confidence=infer_config.get("confidence"),
                use_origin_when_no_meter=parse_bool(infer_config.get("use_origin_when_no_meter", False)),
                start_end_distance_threshold=infer_config.get("start_end_distance_threshold"),
                start_end_position=infer_config.get("start_end_position", "start_left_end_right"),
                snap_out_of_range_pointer=parse_bool(infer_config.get("snap_out_of_range_pointer", True)),
                correction_mode=correction_mode,
                visualization_mode="debug",
                stretch_x_ratio=infer_config.get("stretch_x_ratio", 1.0),
                stretch_y_ratio=infer_config.get("stretch_y_ratio", 1.0),
                reading_offset=infer_config.get("reading_offset", 0.0),
                default_start_angle=infer_config.get("default_start_angle", 45.0),
                default_range_angle=infer_config.get("default_range_angle", 270.0),
                validate_mask_line=parse_bool(infer_config.get("validate_mask_line", True)),
                mask_center_threshold_ratio=infer_config.get("mask_center_threshold_ratio", 0.10),
                reading_backend="compare",
                geometry_fallback_to_transformer=parse_bool(infer_config.get("geometry_fallback_to_transformer", False)),
            )
            reading_details = getattr(self.model, "_last_reading_details", None) or {}
            error = None
            if corpImg is None:
                error = getattr(self.model, "last_error_message", None) or "推理失败"
            elif parse_bool(infer_config.get("auto_zero", True)):
                threshold = infer_config.get("auto_zero_threshold", 0.025)
                try:
                    threshold = float(threshold)
                except (TypeError, ValueError):
                    threshold = 0.025
                if resultNum < threshold:
                    resultNum = 0.0
                    if reading_details.get("selected"):
                        reading_details["selected"]["resultNum"] = resultNum
            return endNum, resultNum, segPointer, corpImg, origin_crop, error, reading_details
        except Exception as exc:
            logger.exception(f"推理失败: {image_path}, correction_mode={correction_mode}")
            return None, None, None, None, None, str(exc), None

    def infer_all_modes(self, image_path, config):
        results = {}
        for mode_key, _ in CORRECTION_MODES:
            cache_key = (image_path, mode_key, self._config_hash(config, mode_key))
            if cache_key not in self.cache:
                self.cache[cache_key] = self.infer_one(image_path, config, mode_key)
            results[mode_key] = self.cache[cache_key]
        return results

    def clear_rows(self):
        for child in self.rows_container.winfo_children():
            child.destroy()
        self.photo_refs = []

    def add_result_row(self, row_idx, image_path, mode_key, mode_label, result):
        endNum, resultNum, segPointer, corpImg, _, error, reading_details = result
        row = ttk.Frame(self.rows_container, padding=4)
        row.grid(row=row_idx, column=0, sticky="w")

        status = f"{mode_label}"
        if error:
            status += f" | {error}"
        else:
            status += f" | selected endNum={endNum} | result={resultNum:.4f}"
        ttk.Label(row, text=status, width=100, font=("Microsoft YaHei", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")

        detail_lines = []
        if reading_details:
            selected_backend = reading_details.get("selected_backend", "transformer")
            branch = reading_details.get("branch", "N/A")
            detail_lines.append(f"selected_backend={selected_backend} | branch={branch}")
            transformer = reading_details.get("transformer") or {}
            geometry = reading_details.get("geometry") or {}
            transformer_result = transformer.get("resultNum")
            geometry_result = geometry.get("resultNum")
            transformer_text = "N/A" if transformer_result is None else f"{transformer_result:.4f}"
            geometry_text = "N/A" if geometry_result is None else f"{geometry_result:.4f}"
            detail_lines.append(
                f"transformer: endNum={transformer.get('endNum', 'N/A')} result={transformer_text} status={transformer.get('status', False)}"
            )
            detail_lines.append(
                f"geometry: endNum={geometry.get('endNum', 'N/A')} result={geometry_text} status={geometry.get('status', False)}"
            )
            if transformer_result is not None and geometry_result is not None:
                detail_lines.append(f"delta(geometry-transformer)={geometry_result - transformer_result:+.4f}")
            geometry_message = geometry.get("message")
            if geometry_message and geometry_message != "OK":
                detail_lines.append(f"geometry_msg={geometry_message}")
        else:
            detail_lines.append("no reading_details")

        ttk.Label(
            row,
            text="\n".join(detail_lines),
            width=100,
            justify="left",
            foreground="#444"
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 6))

        images = [
            ("原图", read_image(image_path)),
            ("结果图", corpImg),
            ("掩码图", segPointer),
        ]
        for col, (label, image) in enumerate(images):
            frame = ttk.Frame(row)
            frame.grid(row=2, column=col, padx=4, pady=2)
            ttk.Label(frame, text=label).pack(anchor="w")
            photo = cv_image_to_photo(image, self.args.cell_width, self.args.cell_height)
            self.photo_refs.append(photo)
            ttk.Label(frame, image=photo).pack()

    def render_current_image(self):
        config = self._read_vars_config()
        self.per_index_config[str(self.index)] = config
        image_path = self.image_paths[self.index]
        current_rel = os.path.relpath(image_path, self.image_root).replace("\\", "/")
        saved_rel = ""
        if self.index < len(self.saved_image_order):
            saved_rel = self.saved_image_order[self.index].get("relative_path", "")
        self.header_var.set(f"[{self.index + 1}/{len(self.image_paths)}] 当前: {current_rel}    保存顺序: {saved_rel}")
        self.status_var.set("推理中...")
        self.root.update_idletasks()

        results = self.infer_all_modes(image_path, config)
        self.clear_rows()
        for row_idx, (mode_key, mode_label) in enumerate(CORRECTION_MODES):
            self.add_result_row(row_idx, image_path, mode_key, mode_label, results[mode_key])
        self.status_var.set(f"配置文件: {self.config_path}")

    def apply_and_rerun(self):
        self.per_index_config[str(self.index)] = self._read_vars_config()
        self.render_current_image()

    def prev_image(self):
        self.per_index_config[str(self.index)] = self._read_vars_config()
        self.index = (self.index - 1) % len(self.image_paths)
        self._load_config_to_vars(self._get_current_config())
        self.render_current_image()

    def next_image(self):
        self.per_index_config[str(self.index)] = self._read_vars_config()
        self.index = (self.index + 1) % len(self.image_paths)
        self._load_config_to_vars(self._get_current_config())
        self.render_current_image()

    def copy_current_to_all(self):
        config = self._read_vars_config()
        for idx in range(len(self.image_paths)):
            self.per_index_config[str(idx)] = config.copy()
        messagebox.showinfo("完成", "已复制当前配置到全部图片序号")

    def _config_payload(self):
        self.per_index_config[str(self.index)] = self._read_vars_config()
        return {
            "version": 1,
            "image_order": self._get_image_order(),
            "global_config": self.global_config,
            "per_index_config": self.per_index_config,
            "browser_options": {
                "correction_modes": [mode for mode, _ in CORRECTION_MODES],
            },
        }

    def save_config(self):
        payload = self._config_payload()
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.saved_image_order = payload["image_order"]
        self.status_var.set(f"已保存: {self.config_path}")

    def save_config_as(self):
        path = filedialog.asksaveasfilename(
            title="保存 inferconfig",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self.config_path = path
        self.save_config()

    def load_config(self, silent=False):
        path = self.config_path
        if not silent:
            selected = filedialog.askopenfilename(
                title="读取 inferconfig",
                initialdir=os.path.dirname(path),
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            )
            if not selected:
                return
            path = selected
            self.config_path = path

        if not os.path.exists(path):
            if not silent:
                messagebox.showwarning("未找到配置", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.global_config = DEFAULT_INFER_CONFIG.copy()
        self.global_config.update(payload.get("global_config", {}))
        self.per_index_config = {
            str(key): value
            for key, value in payload.get("per_index_config", {}).items()
        }
        self.saved_image_order = payload.get("image_order", [])
        self._load_config_to_vars(self._get_current_config())
        if not silent:
            self.render_current_image()


def parse_args():
    parser = argparse.ArgumentParser(description="指针表计识别结果 GUI 调参工具。")
    parser.add_argument("image_root", help="待查看图片目录")
    parser.add_argument("--recursive", action="store_true", help="递归读取 image_root 下的图像")
    parser.add_argument("--device", default=None, help="运行设备，默认自动选择 cuda/cpu")
    parser.add_argument("--config", default=None, help="inferconfig 保存/读取路径")
    parser.add_argument("--cell-width", type=int, default=320, help="单个预览图最大宽度")
    parser.add_argument("--cell-height", type=int, default=220, help="单个预览图最大高度")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.image_root):
        raise FileNotFoundError(f"image_root 不存在: {args.image_root}")

    root = tk.Tk()
    app = QuickBrowseApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
