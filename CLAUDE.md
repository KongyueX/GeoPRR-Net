# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

### Run the API server
```bash
python main.py
```
- Starts the FastAPI app using `config/app_config.yaml` for `host` and `port`.
- This is the preferred entrypoint because `main.py` also starts the config monitor thread and routes uvicorn logging through Loguru.

### Run a lightweight endpoint smoke test
```bash
python -c "import client_test; client_test.test_root_endpoint()"
python -c "import client_test; client_test.test_items_endpoint()"
```
- `client_test.py` is an ad-hoc integration script that assumes the server is already running at `http://127.0.0.1:30600`.

### Run the current integration script
```bash
python client_test.py --smoke-only
python client_test.py --image <image_path>
```
- The API server must already be running. `--smoke-only` checks lightweight endpoints; `--image` additionally exercises `/infer`.

### Run the direct model smoke script
```bash
python test/test.py <image_path> --device cpu
```
- This bypasses FastAPI and instantiates `meterZeroShot` directly.
- Use `--scale-start`, `--scale-end`, and `--device` to match the target meter.

### Batch inference to Excel
```bash
python batch_infer_260520_to_excel.py --image-dir <image_dir> --output <output.xlsx> --result-image-dir <result_image_dir>
```
- Calls `/infer` for each image and writes an Excel report with original images plus returned pointer/mask previews.

### GUI parameter tuning tool
```bash
python utils/angleDetect/quick_browse_results.py <image_root>
python utils/angleDetect/quick_browse_results.py <image_root> --config <config_json_path>
```
- Tkinter-based comparison tool for trying different correction modes and editing `inferConfig` values interactively.

### Research/training environment
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/requirements-training.lock.txt
python -m unittest test.test_selective_geometry
```
- The paper pipeline and pinned CUDA-capable environment are documented in `experiments/README.md`.
- There is no project-level `pyproject.toml`, service dependency lock, lint config, or packaging/build command yet; `experiments/requirements-training.lock.txt` covers the reproducible research workflow.

## High-level architecture

### Service shape
This repository is a FastAPI service for pointer gauge reading. The main production path is:
1. Receive an `/infer` request with `modelName`, `inferData`, `dataType`, and optional `inferConfig`.
2. Convert the input into an OpenCV image via `services/DataProcessService.py`.
3. Dynamically load the requested model from `models/model_list.yaml`.
4. Reuse a singleton model instance from the global `model_instances` cache in `main.py`.
5. Run inference and return a JSON result, optionally including base64-encoded pointer and mask images.

### Entry point and request routing
- `main.py` owns the FastAPI app, request logging middleware, endpoint definitions, config readers, and the global model cache.
- The important endpoints are:
  - `POST /infer`: one-shot inference
  - `POST /beginCaptureWithInfer`: start continuous camera capture + inference
  - `POST /inferWithCapturing`: read the latest queued inference result from an active capture session
  - `POST /endCapture`: stop a capture session
- `main.py` also defines `GET /`, `GET /items/{item_id}`, and `GET /infer_test/{modelName}/{inferData}` (a debug inference route with its own code path, separate from `/infer`). These are template/leftover routes rather than part of the production contract.
- Static Swagger/ReDoc assets are served from `static/` and mounted at `/static`.

### Configuration model
- `config/app_config.yaml` contains server settings (`host`, `port`, `workers`), but `main.py` only reads `host`/`port` for uvicorn — the `workers` value is never passed to `uvicorn.Config(...)` and has no effect.
- `main.py` reads `app_config.yaml` on startup for uvicorn configuration.
- `ConfigMonitor` in `main.py` watches `models/model_list.yaml` for hot reload behavior; it does **not** hot-reload `app_config.yaml` into a running server.

### Dynamic model loading
- `models/model_list.yaml` is the dispatch table from request `modelName` to Python class.
- In practice, the fields that matter are `class_file` and `class_name`; the `path` fields are descriptive leftovers and are not used by the loader.
- `models/BaseInferModel.py` implements a per-subclass singleton pattern with locking, so each model class is loaded once and reused.
- `PointerMeterInferModel` is the real production model; `MyInferModel` is a simple template/test implementation.

### Image acquisition and camera handling
- `services/DataProcessService.py` is the input normalization layer.
  - `filepath` → `cv2.imread`
  - `base64` → base64 decode + `cv2.imdecode`
  - `cameraId` → delegated to `CameraProcessService`
- `services/CameraProcessService.py` isolates camera reads in separate multiprocessing workers.
  - Each camera gets its own process and bounded frame queue.
  - `DataProcessService` lazily starts capture on first request for a camera.
  - Continuous inference mode in `main.py` adds another thread layer (`beginCaptureWithInfer`) that repeatedly pulls frames and queues inference results.
- There are two state registries to be aware of:
  - `DataProcessService.isCapturingDict` tracks whether capture has been started
  - `main.py:capture_thread_dict` tracks continuous inference threads for `/beginCaptureWithInfer`

### Inference pipeline


The production pipeline is split across `models/PointerMeterInferModel.py` and `utils/angleDetect/zeroShotMeter.py`.

`PointerMeterInferModel` is the HTTP-facing adapter:
- Parses `inferConfig`
- Normalizes multiple config aliases from the README
- Serializes optional result images back to base64
- Guards inference with `_infer_lock` so shared model state is used serially

`meterZeroShot` in `utils/angleDetect/zeroShotMeter.py` is the actual CV pipeline:
1. Detect the meter crop with YOLO
2. Optionally correct the crop (`off`, `ransacFun`, `ransacFunbackup`, `square`, `stretch`, `ellipse`) — `ellipse` is README's recommended default for tilted meters and is implemented via `_ellipse_rectify`
3. Segment the pointer mask with U2Net
4. Validate the mask axis against the dial center when `validate_mask_line` is enabled (returns early with a "无法找到指针" failure if invalid)
5. Conditionally run the transformer-based meter model to estimate the normalized pointer position
6. Detect start/end reference points with another YOLO model
7. Compute the final reading. This is not a single formula: the pipeline computes multiple reading-backend candidates (`transformer`, `geometry`/legacy, `geometry_direct`, `geometry_direct_v2`, `geometry_fusion`, `geometry_fusion_calibrated`, `geometry_hybrid`, `geometry_hybrid_gate`), each with its own fallback chain in `_select_reading_output`, and returns whichever one `reading_backend` selects (falling back to `transformer` if the selected backend fails and fallback is enabled).

The weights loaded by `PointerMeterInferModel` come from the checked-in `utils/angleDetect/**/result*/*.pt` files, not from an external registry.

### Important `inferConfig` behavior
The README is the authoritative explanation of request parameters. The implementation in `PointerMeterInferModel` currently supports many alias groups, including:
- `confidence` / `conf`
- `correction_mode` / `image_correction` / `correction_algorithm` (modes: `off`, `ransacFun`, `ransacFunbackup`, `square`, `stretch`, `ellipse`)
- `stretch_x_ratio` / `horizontal_stretch_ratio`
- `stretch_y_ratio` / `vertical_stretch_ratio`
- `reading_offset` / `endNum_offset`
- `validate_mask_line` / `check_mask_line`
- `auto_zero` / `auto_zero_reading`
- `auto_zero_threshold` / `zero_threshold`
- `reading_backend` / `reading_method` / `meter_reading_backend` — selects the reading algorithm (`transformer`, `geometry`, `geometry_direct`, `geometry_direct_v2`, `geometry_fusion`, `geometry_fusion_calibrated`, `geometry_hybrid`, `geometry_hybrid_gate`, `geometry_legacy`, `compare`)
- `geometry_fallback_to_transformer` / `fallback_geometry_to_transformer`
- `residual_calibrator_path` / `geometry_calibrator_path` / `calibration_model_path`
- `residual_hybrid_model_path` / `geometry_hybrid_model_path` / `hybrid_model_path` and `residual_hybrid_max_abs_delta` / `geometry_hybrid_max_abs_delta` / `hybrid_max_abs_delta` (used by the hybrid backends; not documented in README)
- `use_origin_when_no_meter`, with code-only aliases `continue_infer_without_meter` / `infer_origin_when_no_meter` (not documented in README)
- `return_reading_details` / `return_backend_details`
- `start_end_distance_threshold` / `same_point_distance_threshold` / `point_same_distance_threshold`
- `snap_out_of_range_pointer` / `clamp_out_of_range_pointer` / `snap_pointer_when_out_of_range`

This list is not exhaustive — README documents further aliases and standalone params (e.g. `start_end_position`, `default_start_angle`, `default_range_angle`, `mask_center_threshold_ratio`, `result_pointer_image`, `result_mask_image`). When changing request semantics, keep `README.md` and the alias handling in `PointerMeterInferModel` aligned, and be aware the two `geometry_hybrid*` backend params above exist in code but are currently undocumented in README itself.

### Logging
- `config/log.py` configures Loguru sinks under `log/user_log/`.
- Uvicorn loggers are intercepted so API request logs and application logs end up in the same Loguru-managed output.
- The configured filename prefix is currently `PipeStatusBackend-`, which does not match the repository name; treat that as existing project behavior rather than a signal that this is a different service.

### Scripts and developer utilities
- `client_test.py` is not a clean automated test suite; it is a manual integration helper with optional OpenCV preview windows. It contains real `assert` statements but relies on a running server, so it is not pytest-collectible as-is.
- `test/test.py` is a direct local debug script for the model pipeline; it has no assertions, runs only from `__main__`, and requires an explicit image path.
- `batch_infer_260520_to_excel.py` is the most reusable offline utility for bulk API evaluation.
- `utils/angleDetect/quick_browse_results.py` is the main tuning workflow when adjusting correction or reading parameters.
- There is no automated test suite anywhere in the repo (no pytest config, no assertion-based suite that runs headlessly).
- The repo root also has a long tail of one-off experiment/comparison scripts not covered above (e.g. `batch_compare_reading_backends.py`, `batch_compare_u2net_models.py`, `exp_ellipse_rectify*.py`, `exp_ellipse_mask_strategies*.py`, `exp_angle_sensitivity.py`, `verify_with_ground_truth.py`, `verify_with_yolo_center*.py`, `report_data_html.py`, `make_pointer_meter_docx.py`). These are developer scratch tools for tuning/evaluating the reading backends, not part of the production path.

## Working assumptions for future changes
- Most API behavior changes will require reading both `main.py` and `models/PointerMeterInferModel.py`.
- Most reading-accuracy changes will require tracing into `utils/angleDetect/zeroShotMeter.py` and its helper modules under `utils/angleDetect/`.
- Changes to model selection or adding a new model require updating `models/model_list.yaml` and providing a `BaseInferModel` subclass that can be imported dynamically.
- Camera-related issues usually span `main.py`, `services/DataProcessService.py`, and `services/CameraProcessService.py` rather than living in one place.
