#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AEGISLM_MISTRAL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SERVING_ROOT="${PROJECT_ROOT}/serving/vllm"
SOURCE_MODEL="${MISTRAL_NATIVE_MODEL_ROOT:-${PROJECT_ROOT}/model/mistral-small-4-119b-2603}"
LOG="${VLLM_LOG_PATH:-${PROJECT_ROOT}/artifact/mistral-small-4-119b/logs/vllm-native-base-smoke.log}"
CACHE_ROOT="${VLLM_CACHE_ROOT:-${SERVING_ROOT}/.runtime-cache}"

# Compilation cache의 shared object는 executable filesystem에 두고, Unix socket이
# 사용하는 TMPDIR은 sockaddr_un 길이 제한을 피하도록 짧게 유지한다.
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export TMPDIR="${VLLM_TMPDIR:-${HOME}/.vllm-tmp}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${TMPDIR}" \
  "$(dirname "${LOG}")"

cd "${SERVING_ROOT}"

uv run --no-sync vllm serve "$SOURCE_MODEL" \
  --config-format mistral \
  --load-format mistral \
  --tokenizer-mode mistral \
  --served-model-name mistral-native-base-smoke \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --kernel-config '{"enable_cutedsl_warmup": false}' \
  --host 127.0.0.1 \
  --port 8000 \
  2>&1 | tee "$LOG"
