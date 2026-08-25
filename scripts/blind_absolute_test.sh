# test code.
SOURCE_REPO="../AegisLM-B200-phase-f-source-v3"
EVALUATOR="$SOURCE_REPO/scripts/evaluate_source_decision.py"

EVAL_DIR="artifact/mistral-small-4-119b/evaluations/phase-f-source-fresh-blind-500-v1/g3-bf16-lora-fsdp2-20260819-001"

PREDICTIONS="$EVAL_DIR/predictions.jsonl"
GOLD="data/processed/phase-f-source-fresh-blind-contracts-500-v1/decision/gold.jsonl"
SUMMARY="$EVAL_DIR/summary.json"
SCORE_LOG="artifact/mistral-small-4-119b/logs/g3-bf16-lora-fsdp2-20260819-001.blind500-score.log"

test -f "$EVALUATOR"
test -s "$PREDICTIONS"
test -s "$GOLD"
test ! -e "$SUMMARY"
test ! -e "$SCORE_LOG"

set +e
set -o pipefail

uv run --no-sync python "$EVALUATOR" \
  --gold "$GOLD" \
  --predictions "$PREDICTIONS" \
  --summary "$SUMMARY" \
  --minimum-sample-count 500 \
  --minimum-precision 0.90 \
  --minimum-recall 0.95 \
  --maximum-false-positive-rate 0.05 \
  --maximum-abstention-rate 0.05 \
  --minimum-parse-success-rate 0.99 \
  --minimum-schema-pass-rate 0.99 \
  --blind-test \
  --fail-on-gate 2>&1 |
  tee "$SCORE_LOG"

SCORE_EXIT=${PIPESTATUS[0]}

printf 'scorer exit code: %s\n' "$SCORE_EXIT" |
  tee -a "$SCORE_LOG"

# print python code
uv run --no-sync python - "$SUMMARY"<< 'PY'
import json
import sys
from pathlib import Path

summary = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)

print("overall_pass:", summary["overall_pass"])

print("\nmetrics")
for key, value in summary["metrics"].items():
    print(f"  {key}: {value}")

print("\ngates")
for key, value in summary["gates"].items():
    print(f"  {key}: {value}")

PY

# 결과 inventory 보존
RESULT_INVENTORY="$EVAL_DIR/SHA256SUMS"

test ! -e "$RESULT_INVENTORY"

sha256sum \
  "$PREDICTIONS" \
  "$SUMMARY" \
  "$SCORE_LOG" |
  tee "$RESULT_INVENTORY"