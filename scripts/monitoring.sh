#!/usr/bin/env bash

set -u

NEW_ID="g3-bf16-lora-fsdp2-20260819-001"
MONITOR_DIR="artifact/mistral-small-4-119b/monitoring"
MONITOR_CSV="${MONITOR_DIR}/${NEW_ID}.csv"
MONITOR_ERR="${MONITOR_DIR}/${NEW_ID}.stderr.log"

mkdir -p "$MONITOR_DIR"

trap '
  echo
  echo "GPU monitor stopped"
  exit 0
' INT TERM

if [[ ! -e "$MONITOR_CSV" ]]; then
  echo 'timestamp,index,name,memory.used [MiB],utilization.gpu [%]' \
    > "$MONITOR_CSV"
fi

echo "writing GPU telemetry to: $MONITOR_CSV"
echo "press Ctrl+C to stop"

while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    >> "$MONITOR_CSV" \
    2>> "$MONITOR_ERR"

  sleep 1
done