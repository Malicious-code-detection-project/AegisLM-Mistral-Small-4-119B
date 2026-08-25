# AegisLM Mistral Small 4 119B

Private repository: `Malicious-code-detection-project/AegisLM-Mistral-Small-4-119B`

B200 두 장에서 `mistralai/Mistral-Small-4-119B-2603`을 BF16 LoRA + FSDP2로
파인튜닝하고, adapter lifecycle·blind evaluation·vLLM serving을 검증하는 독립
프로젝트다.

## 현재 상태

- 공식 FP8 checkpoint를 local Hugging Face BF16 checkpoint로 변환·검증
- train 10,000건과 validation 1,000건 tokenizer preflight PASS
- bitsandbytes QLoRA는 fused expert tensor의 `INT_MAX + 1` 크기 제한으로 BLOCK
- BF16 LoRA + FSDP2 G1 1-step, G2 10-step save/reload, G3 100-step PASS
- G3 source-decision fresh blind 500건 자동 gate PASS
- official native FP8 base의 vLLM 0.26.0 TP2 Chat Completions smoke PASS
- vLLM에서 G3 LoRA adapter attach는 다음 compatibility gate

## 환경

학습과 서빙 dependency가 충돌하므로 두 uv project를 분리한다.

| 역할 | 위치 | 핵심 runtime |
| --- | --- | --- |
| 학습·평가 | project root | Python 3.12.3, PyTorch 2.12.1+cu130, Axolotl 0.17.0 |
| 서빙 | `serving/vllm/` | Python 3.12.3, vLLM 0.26.0 CUDA 13 wheel |

```bash
uv sync --frozen
uv run --no-sync python scripts/preflight_dataset.py
```

서빙 환경은 별도로 동기화한다.

```bash
cd serving/vllm
uv sync --frozen
cd ../..
bash scripts/check_serving.sh
```

## 저장 경계

이 repository에는 code, config와 dependency lock만 저장한다. 다음 항목은 local 또는
persistent storage에만 둔다.

- `.env`와 인증 token
- model checkpoint와 tokenizer snapshot
- raw/processed dataset
- LoRA adapter, optimizer state와 evaluation artifact
- GPU monitoring·training·serving log
- TorchInductor, Triton과 CUDA compilation cache
- virtual environment

기본 runtime 경로는 repository-relative다. 다른 storage layout을 사용하면 `.env` 또는
shell environment에서 다음 변수를 지정한다.

```bash
export AEGISLM_MISTRAL_ROOT="$(pwd)"
export MISTRAL_NATIVE_MODEL_ROOT="${AEGISLM_MISTRAL_ROOT}/model/mistral-small-4-119b-2603"
```

## 검증

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy scripts/
uv run pytest
```

GPU가 필요한 model load·training·serving 검사는 B200에서 별도로 수행한다.

## Provenance

이 repository의 최초 source snapshot은 2026-08-25에 B200 standalone project에서
model·data·artifact·secret·cache를 제외하고 복제했다. 초기 allowlist 40개 파일은 수정
전 SHA-256이 B200과 모두 일치했다. 자세한 범위와 차이는
[`docs/B200_SYNC_REPORT_20260825.md`](docs/B200_SYNC_REPORT_20260825.md)에 기록한다.
