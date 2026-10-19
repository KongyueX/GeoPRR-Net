#!/usr/bin/env bash
set -e

geoprr_render_root="${GEOPRR_HOME:-/root/autodl-tmp/geoprrnet}/rendering"
export CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH="$geoprr_render_root/tools/sysroot/usr/lib/x86_64-linux-gnu"
export PYTHONPATH="$geoprr_render_root/python_packages"
export PYTHONNOUSERSITE=1
exec "$geoprr_render_root/tools/blender-4.1.1-linux-x64/blender" --python-use-system-env "$@"
