G3_ID="g3-bf16-lora-fsdp2-20260819-001"
G3_LOG="artifact/mistral-small-4-119b/logs/${G3_ID}.train.log"
G3_CSV="artifact/mistral-small-4-119b/monitoring/${G3_ID}.csv"
G3_RUN="artifact/mistral-small-4-119b/runs/${G3_ID}"
G3_CONFIG="configs/g3-bf16-lora-fsdp2-100step-20260819-001.yaml"

LOSS_COUNT="$(
  grep -o "'loss': '[^']*'" "$G3_LOG" |
    wc -l
)"

echo "loss count: $LOSS_COUNT"
test "$LOSS_COUNT" -eq 100 &&
  echo "G3 STEP COUNT PASS"

grep -nEi \
  'train_runtime|train_loss|Model successfully|Traceback|ChildFailedError|out of memory|nan|infinity' \
  "$G3_LOG" |
  tail -n 50

test -s "$G3_RUN/adapter_model.safetensors" &&
  grep -q 'Model successfully saved' "$G3_LOG" &&
  echo "G3 ADAPTER SAVE PASS"

awk -F',' '
NR > 1 {
    gpu=$2+0
    mem=$4+0

    if (mem > peak[gpu])
        peak[gpu]=mem
}
END {
    failed=0

    for (gpu=0; gpu<=1; gpu++) {
        printf "GPU %d peak: %d MiB\n", gpu, peak[gpu]

        if (peak[gpu] > 168960) {
            printf "GPU %d VRAM GATE FAIL\n", gpu
            failed=1
        }
    }

    exit failed
}' "$G3_CSV" &&
  echo "G3 VRAM GATE PASS"

G3_INVENTORY="artifact/mistral-small-4-119b/logs/${G3_ID}.sha256"

test ! -e "$G3_INVENTORY"

sha256sum \
  "$G3_CONFIG" \
  "$G3_RUN/adapter_model.safetensors" \
  "$G3_RUN/adapter_config.json" \
  "$G3_RUN/config.json" \
  "$G3_RUN/tokenizer_config.json" \
  "$G3_RUN/processor_config.json" |
  tee "$G3_INVENTORY"

test -s "$G3_INVENTORY" &&
  echo "G3 ARTIFACT INVENTORY PASS"
