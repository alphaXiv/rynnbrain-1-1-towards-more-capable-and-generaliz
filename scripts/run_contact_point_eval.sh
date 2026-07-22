#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir -q -r requirements.txt
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

echo "RUN_CONFIG_BEGIN"
cat config.json
echo "RUN_CONFIG_END"
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu_count={torch.cuda.device_count()}")
assert torch.cuda.device_count() == 8, "benchmark contract requires exactly 8 visible GPUs"
PY

python scripts/evaluate_contact_points_tp.py
