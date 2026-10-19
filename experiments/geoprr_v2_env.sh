#!/usr/bin/env bash
# Source this file before preparing data or running the lightweight reader.
export GEOPRR_HOME="${GEOPRR_HOME:-/root/autodl-tmp/geoprrnet}"
export GEOPRR_CODE="$GEOPRR_HOME/code"
source "$GEOPRR_HOME/.venv/bin/activate"
export HF_HOME="$GEOPRR_HOME/cache/huggingface"
export TORCH_HOME="$GEOPRR_HOME/cache/torch"
# Both selected public checkpoints are cached on this server.
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$GEOPRR_CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$TORCH_HOME"
