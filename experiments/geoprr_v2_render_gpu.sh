#!/usr/bin/env bash
set -e
rendering_root="${GEOPRR_HOME:-/root/autodl-tmp/geoprrnet}/rendering"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$rendering_root/tools/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$rendering_root/python_packages${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
exec "$rendering_root/tools/blender-4.5.14-linux-x64/blender" --python-use-system-env "$@"
