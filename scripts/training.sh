# G2_CONFIG="configs/g2-bf16-lora-fsdp2-10step-20260818-001.yaml"
# NEW_ID="g2-bf16-lora-fsdp2-20260818-001"
G3_CONFIG="configs/g3-bf16-lora-fsdp2-100step-20260819-001.yaml"
NEW_ID="g3-bf16-lora-fsdp2-20260819-001"
TRAIN_LOG="artifact/mistral-small-4-119b/logs/${NEW_ID}.train.log"

test ! -e "$TRAIN_LOG"

export CUDA_VISIBLE_DEVICES=0,1
export AXOLOTL_DO_NOT_TRACK=1
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_USE_FSDP=true
export ACCELERATE_MIXED_PRECISION=bf16

set -o pipefail

uv run --no-sync torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  -m axolotl.cli.train \
  "$G3_CONFIG" 2>&1 | tee "$TRAIN_LOG"
#   "$G2_CONFIG" 2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "training exit code: $TRAIN_EXIT"