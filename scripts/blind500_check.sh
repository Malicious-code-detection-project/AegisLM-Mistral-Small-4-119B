echo "=== Blind datasets ==="

find data/processed -maxdepth 3 -type f |
  grep -Ei 'blind|challenge|gold|500' |
  sort

echo "=== Evaluation scripts ==="

find scripts -maxdepth 3 -type f |
  grep -Ei 'eval|predict|blind|inference|score' |
  sort

echo "=== Evaluator references ==="

rg -n \
  'phase-f-source-fresh-blind-500-v1|blind.?500|evaluate_predictions|precision|abstention' \
  scripts configs pyproject.toml 2>/dev/null || true