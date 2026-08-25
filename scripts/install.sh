#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="$PROJECT_ROOT/.env"
MODEL_DIR="$PROJECT_ROOT/model/mistral-small-4-119b-2603"
BF16_MODEL_DIR="$PROJECT_ROOT/model/mistral-small-4-119b-2603-bf16-local"
ACTION="${1:-check}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env 파일이 없습니다: $ENV_FILE" >&2
    exit 1
fi

# .env의 값을 이 스크립트와 자식 프로세스에 export한다.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ ! "${MISTRAL_OFFICIAL_REVISION:-}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: MISTRAL_OFFICIAL_REVISION은 40자리 SHA여야 합니다." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv 명령을 찾을 수 없습니다." >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

DOWNLOAD_ARGS=(
    mistralai/Mistral-Small-4-119B-2603
    --revision "$MISTRAL_OFFICIAL_REVISION"
    --local-dir "$MODEL_DIR"
    --include "model*.safetensors"
    --include "*.json"
    --include "*.jinja"
    --include "*.txt"
    --include "README.md"
)

check_environment() {
    echo "Project: $PROJECT_ROOT"
    echo "Revision: ${MISTRAL_OFFICIAL_REVISION:0:12}..."
    echo "Model directory: $MODEL_DIR"

    uv run python --version
    uv run hf auth whoami

    echo
    echo "Model filesystem:"
    df -h "$PROJECT_ROOT/model"
}

case "$ACTION" in
    check)
        check_environment
        ;;

    sync)
        check_environment
        uv sync --locked
        ;;

    dry-run)
        check_environment
        uv run hf download "${DOWNLOAD_ARGS[@]}" --dry-run
        ;;

    download)
        check_environment
        uv run hf download "${DOWNLOAD_ARGS[@]}"

        test -f "$MODEL_DIR/config.json"
        test -f "$MODEL_DIR/model.safetensors.index.json"

        echo
        echo "Download completed."
        du -sh "$MODEL_DIR"

        find "$MODEL_DIR" \
            -maxdepth 1 \
            -name 'model-*.safetensors' \
            -printf '%f\n' \
            | sort
        ;;

    convert)
        check_environment

        uv run python scripts/convert_official_fp8_to_bf16.py \
            --source "$MODEL_DIR" \
            --output "$BF16_MODEL_DIR" \
            --revision "$MISTRAL_OFFICIAL_REVISION" \
            --max-shard-size "16GB"
        ;;

    verify-bf16)
        test -f "$BF16_MODEL_DIR/config.json"
        test -f "$BF16_MODEL_DIR/model.safetensors.index.json"

        uv run python - <<PY
import json
from pathlib import Path

root = Path("$BF16_MODEL_DIR")
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json).read_text())
manifest = json.loads((root / "conversion_manifest.json).read_text())

assert "quantization_config" not in config
assert "quantization_config" not in config.get("text_config", {})
assert config["dtype"] == "bfloat16"

print("dtype:", config["dtype"])
print("tensor count:", len(index["weight_map"]))
print("shard count:", len(set(index["weight_map"].values())))
print("stored bytes:", index["metadata"]["total_size"])
print("source revision:", manifest["source_revision"])
print("BF16 checkpoint verification PASS)
PY
        du -sh "$BF16_MODEL_DIR"
        ;;

    *)
        echo "Usage: $0 {check|sync|dry-run|download|convert|verify-bf16}" >&2
        exit 2
        ;;
esac