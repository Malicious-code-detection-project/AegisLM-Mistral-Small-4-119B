PREDICT_LOG="artifact/mistral-small-4-119b/logs/g3-bf16-lora-fsdp2-20260819-001.blind500-predict.log"

test ! -e "$PREDICT_LOG"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set -o pipefail

uv run --no-sync python -B -u scripts/predict_blind500.py 2>&1 |
  tee "$PREDICT_LOG"

PREDICT_EXIT=${PIPESTATUS[0]}

printf 'prediction exit code: %s\n' "$PREDICT_EXIT" |
  tee -a "$PREDICT_LOG"