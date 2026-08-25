BLIND_ROOT="data/processed/phase-f-source-fresh-blind-500-v1"
CONTRACT_ROOT="data/processed/phase-f-source-fresh-blind-contracts-500-v1"

wc -l \
  "$BLIND_ROOT/challenge.jsonl" \
  "$BLIND_ROOT/gold.jsonl" \
  "$CONTRACT_ROOT/decision/challenge.jsonl" \
  "$CONTRACT_ROOT/decision/gold.jsonl"

(
  cd "$BLIND_ROOT"
  sha256sum -c SHA256SUMS
)

(
  cd "$CONTRACT_ROOT"
  sha256sum -c SHA256SUMS
)

uv run --no-sync python scripts/blind_test_challenge_gold.py

sha256sum scripts/blind500_check.sh
sed -n '1,320p' scripts/blind500_check.sh
