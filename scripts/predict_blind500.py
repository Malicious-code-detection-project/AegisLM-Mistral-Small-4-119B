from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

BASE_ROOT = Path("model/mistral-small-4-119b-2603-bf16-local")
ADAPTER_ROOT = Path(
    "artifact/mistral-small-4-119b/runs/"
    "g3-bf16-lora-fsdp2-20260819-001"
)
CHALLENGE_PATH = Path(
    "data/processed/"
    "phase-f-source-fresh-blind-contracts-500-v1/"
    "decision/challenge.jsonl"
)
EVALUATION_ROOT = Path(
    "artifact/mistral-small-4-119b/evaluations/"
    "phase-f-source-fresh-blind-500-v1/"
    "g3-bf16-lora-fsdp2-20260819-001"
)

RUN_ID = "g3-bf16-lora-fsdp2-20260819-001"
MODEL_ID = "mistral-small-4-119b-2603-bf16-local+g3-lora"
EXPECTED_ROLES = ["system", "user"]
FORBIDDEN_MARKERS = ("[THINK]", "[/THINK]")


def load_challenge() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with CHALLENGE_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            if set(row) != {"id", "messages"}:
                raise RuntimeError(
                    f"challenge:{line_number}: unexpected keys"
                )

            messages = row["messages"]
            if not isinstance(messages, list):
                raise TypeError(
                    f"challenge:{line_number}: messages not list"
                )

            roles = [message.get("role") for message in messages]
            if roles != EXPECTED_ROLES:
                raise RuntimeError(
                    f"challenge:{line_number}: roles={roles}"
                )

            for message in messages:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise TypeError(
                        f"challenge:{line_number}: invalid content"
                    )
                if any(marker in content for marker in FORBIDDEN_MARKERS):
                    raise RuntimeError(
                        f"challenge:{line_number}: THINK marker"
                    )

            rows.append(row)

    if len(rows) != 500:
        raise RuntimeError(f"challenge count is {len(rows)}, expected 500")

    record_ids = [str(row["id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("challenge contains duplicate IDs")

    return rows


def load_completed_ids(
    partial_path: Path,
    challenge: list[dict[str, Any]],
) -> list[str]:
    if not partial_path.exists():
        return []

    completed: list[str] = []

    with partial_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)

            if item.get("model_id") != MODEL_ID:
                raise RuntimeError(
                    f"partial:{line_number}: model ID mismatch"
                )
            if item.get("run_id") != RUN_ID:
                raise RuntimeError(
                    f"partial:{line_number}: run ID mismatch"
                )

            completed.append(str(item["record_id"]))

    expected_prefix = [
        str(row["id"]) for row in challenge[: len(completed)]
    ]
    if completed != expected_prefix:
        raise RuntimeError("partial predictions are not a challenge prefix")

    return completed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def synchronize_gpus() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def main() -> None:
    torch.manual_seed(20260728)
    torch.cuda.manual_seed_all(20260728)
    torch.set_grad_enabled(False)

    challenge = load_challenge()
    print("challenge rows:", len(challenge))

    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)

    prediction_path = EVALUATION_ROOT / "predictions.jsonl"
    partial_path = EVALUATION_ROOT / "predictions.inprogress.jsonl"
    hash_path = EVALUATION_ROOT / "predictions.sha256"

    if prediction_path.exists() or hash_path.exists():
        raise RuntimeError("final prediction artifact already exists")

    completed_ids = load_completed_ids(partial_path, challenge)
    start_index = len(completed_ids)

    print("completed rows:", start_index)
    print("remaining rows:", len(challenge) - start_index)

    print("[load] processor")

    processor = AutoProcessor.from_pretrained(
        BASE_ROOT,
        local_files_only=True,
    )
    tokenizer = processor.tokenizer

    print("[load] BF16 base model")

    base_model = Mistral3ForConditionalGeneration.from_pretrained(
        BASE_ROOT,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={
            0: "140GiB",
            1: "140GiB",
            "cpu": "1500GiB",
        },
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )

    placements = {str(value) for value in base_model.hf_device_map.values()}
    if "cpu" in placements or "disk" in placements:
        raise RuntimeError(
            f"unexpected model offload placement: {placements}"
        )

    print("[load] G3 adapter")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ROOT,
        is_trainable=False,
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("active adapters:", model.active_adapters)

    input_device = model.get_input_embeddings().weight.device
    file_mode = "a" if partial_path.exists() else "x"

    with partial_path.open(file_mode, encoding="utf-8") as output:
        for index, row in enumerate(
            challenge[start_index:],
            start=start_index,
        ):
            messages = row["messages"]

            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="none",
            )

            if any(marker in rendered for marker in FORBIDDEN_MARKERS):
                raise RuntimeError(
                    f"{row['id']}: rendered THINK marker"
                )

            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {
                key: value.to(input_device)
                for key, value in inputs.items()
            }

            synchronize_gpus()
            started = time.perf_counter()

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            synchronize_gpus()
            latency_ms = (time.perf_counter() - started) * 1000

            prompt_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0, prompt_length:]
            raw_output = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            prediction = {
                "record_id": str(row["id"]),
                "model_id": MODEL_ID,
                "run_id": RUN_ID,
                "raw_output": raw_output,
                "latency_ms": round(latency_ms, 4),
                "generated_at": datetime.now(UTC).isoformat(),
                "metadata": {
                    "prompt_tokens": int(prompt_length),
                    "generated_tokens": int(generated_ids.numel()),
                    "reasoning_effort": "none",
                    "do_sample": False,
                },
            }

            output.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            output.flush()

            completed_count = index + 1
            if completed_count % 10 == 0:
                os.fsync(output.fileno())
                print(
                    f"[progress] {completed_count}/500 "
                    f"latency_ms={latency_ms:.1f}"
                )

    line_count = sum(
        1
        for line in partial_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if line_count != 500:
        raise RuntimeError(
            f"prediction count is {line_count}, expected 500"
        )

    partial_path.replace(prediction_path)

    digest = sha256_file(prediction_path)
    with hash_path.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {prediction_path.name}\n")

    print("prediction path:", prediction_path)
    print("prediction sha256:", digest)
    print("BLIND500 PREDICTION FREEZE PASS")


if __name__ == "__main__":
    main()