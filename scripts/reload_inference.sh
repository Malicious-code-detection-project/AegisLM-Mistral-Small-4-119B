RELOAD_LOG="artifact/mistral-small-4-119b/logs/g2-bf16-lora-fsdp2-20260818-001.reload.log"

test ! -e "$RELOAD_LOG"

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set -o pipefail

uv run --no-sync python -B -u scripts/reload_inference.py 2>&1 | tee "$RELOAD_LOG"

RELOAD_EXIT=${PIPESTATUS[0]}
echo "reload exit code: $RELOAD_EXIT"