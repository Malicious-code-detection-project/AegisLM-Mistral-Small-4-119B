EVAL_DIR="artifact/mistral-small-4-119b/evaluations/phase-f-source-fresh-blind-500-v1/g3-bf16-lora-fsdp2-20260819-001"

wc -l "$EVAL_DIR/predictions.jsonl"

(
  cd "$EVAL_DIR"
  sha256sum -c predictions.sha256
)

grep -E \
  'BLIND500 PREDICTION FREEZE PASS|prediction exit code' \
  "$PREDICT_LOG"