# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
- Static Swagger/ReDoc assets are served from `static/` and mounted at `/static`.

### Configuration model
- `config/app_config.yaml` only contains server settings (`host`, `port`, `workers`).
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
2. Optionally correct the crop (`off`, `ransacFun`, `ransacFunbackup`, `square`, `stretch`)
3. Segment the pointer mask with U2Net
4. Validate the mask axis against the dial center when `validate_mask_line` is enabled
5. Use the transformer-based meter model to estimate the normalized pointer position
6. Detect start/end reference points with another YOLO model
7. Convert the predicted position into the final reading using `scaleStart` / `scaleEnd`

The weights loaded by `PointerMeterInferModel` come from the checked-in `utils/angleDetect/**/result*/*.pt` files, not from an external registry.

### Important `inferConfig` behavior
The README is the authoritative explanation of request parameters. The implementation in `PointerMeterInferModel` currently supports important alias groups, including:
- `confidence` / `conf`
- `correction_mode` / `image_correction` / `correction_algorithm`
- `stretch_x_ratio` / `horizontal_stretch_ratio`
- `stretch_y_ratio` / `vertical_stretch_ratio`
- `reading_offset` / `endNum_offset`
- `validate_mask_line` / `check_mask_line`
- `auto_zero` / `auto_zero_reading`
- `auto_zero_threshold` / `zero_threshold`

When changing request semantics, keep `README.md` and the alias handling in `PointerMeterInferModel` aligned.

### Logging
- `config/log.py` configures Loguru sinks under `log/user_log/`.
- Uvicorn loggers are intercepted so API request logs and application logs end up in the same Loguru-managed output.
- The configured filename prefix is currently `PipeStatusBackend-`, which does not match the repository name; treat that as existing project behavior rather than a signal that this is a different service.

### Scripts and developer utilities
- `client_test.py` is not a clean automated test suite; it is a manual integration helper that requires a running server and optionally opens OpenCV preview windows.
- `test/test.py` is a direct local debug script for the model pipeline and requires an explicit input image.
- `batch_infer_260520_to_excel.py` is the most reusable offline utility for bulk API evaluation.
- `utils/angleDetect/quick_browse_results.py` is the main tuning workflow when adjusting correction or reading parameters.

## Working assumptions for future changes
- Most API behavior changes will require reading both `main.py` and `models/PointerMeterInferModel.py`.
- Most reading-accuracy changes will require tracing into `utils/angleDetect/zeroShotMeter.py` and its helper modules under `utils/angleDetect/`.
- Changes to model selection or adding a new model require updating `models/model_list.yaml` and providing a `BaseInferModel` subclass that can be imported dynamically.
- Camera-related issues usually span `main.py`, `services/DataProcessService.py`, and `services/CameraProcessService.py` rather than living in one place.
