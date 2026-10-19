#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${1:-/root/autodl-tmp/geoprrnet/.venv}"
bootstrap_python="${BOOTSTRAP_PYTHON:-/root/miniconda3/bin/python}"

if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$bootstrap_python" -m venv "$venv_dir"
fi
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r "$repo_dir/experiments/requirements-geoprr-v2.txt"
"$venv_dir/bin/python" -m pip check
"$venv_dir/bin/python" - <<'PY'
import json
import platform
import sys

import cv2
import timm
import torch
import torchvision

print(json.dumps({
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "environment_prefix": sys.prefix,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "opencv": cv2.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "package_paths": {"torch": torch.__file__, "torchvision": torchvision.__file__,
                      "timm": timm.__file__, "opencv": cv2.__file__},
}, indent=2))
PY
