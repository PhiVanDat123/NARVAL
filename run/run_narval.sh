#!/bin/bash
# Launch NARVAL training with accelerate + DeepSpeed ZeRO-3.
# Run from inside the narval/ folder:  cd narval && bash run/run_narval.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
PYTHONPATH="$HERE:${PYTHONPATH:-}" \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_narval.py \
  recipes/configs/narval.yaml
