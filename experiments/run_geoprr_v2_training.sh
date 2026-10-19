#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_dir/experiments/geoprr_v2_env.sh"
cd "$repo_dir"
manifest_dir="$GEOPRR_HOME/runs/geoprr_v2/manifests"

python -m experiments.prepare_geoprr_v2_data \
    --root "$GEOPRR_HOME/datasets/SyncG/syncG" --output-dir "$manifest_dir"
python -m experiments.audit_geoprr_v2_geometry \
    --manifest "$manifest_dir/source_train.jsonl" --dataset-root "$GEOPRR_HOME/datasets" \
    --image-size 640 --min-support-mass 0.15 \
    --output "$GEOPRR_HOME/runs/geoprr_v2/geometry_audit.json"

for backbone in femto mobilenetv4; do
    python -m experiments.train_geoprr_v2 \
        --config "experiments/configs/geoprr_v2_${backbone}.json" \
        --train-manifest "$manifest_dir/train.jsonl" \
        --val-manifest "$manifest_dir/validation.jsonl"
done
