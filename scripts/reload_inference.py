from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

BASE_ROOT = Path("model/mistral-small-4-119b-2603-bf16-local")
ADAPTER_ROOT = Path(
    "artifact/mistral-small-4-119b/runs/"
    "g2-bf16-lora-fsdp2-20260818-001"
)
VALIDATION_PATH = Path(
    "data/processed/phase-f-source-decision-v1/validation.jsonl"
)

def main() -> None :
    print("[1/5] Loading validation prompt")

    with VALIDATION_PATH.open(encoding="utf-8") as handle :
        record = json.loads(handle.readline())

    messages = record["messages"]
    prompt_messages = messages[:-1]
    expected = messages[-1]["content"]

    print("record id:", record["id"])
    print("expected:", expected)

    print("[2/5] Loading local processor")
    processor = AutoProcessor.from_pretrained(
        BASE_ROOT,
        local_files_only=True,
    )

    tokenizer = processor.tokenizer

    rendered = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="none",
    )

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )

    print("prompt tokens:", inputs["input_ids"].shape[-1])

    print("[3/5] Loading BF16 base model across two GPUs")

    base_model = Mistral3ForConditionalGeneration.from_pretrained(
        BASE_ROOT,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={
            0:"140GiB",
            1:"140GiB",
            "cpu":"1500GiB",
        },
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )

    device_map = base_model.hf_device_map
    print("device map:", device_map)

    placements = {str(device) for device in device_map.values()}
    if "disk" in placements:
        raise RuntimeError("Model was unexpectedly offloaded to disk")

    print("[4/5] Loading final LoRA adapter")

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
    inputs = {
        key:value.to(input_device)
        for key, value in inputs.items()
    }

    print("[5/5] Runnig deterministic inference")
    with torch.inference_mode() :
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0, prompt_length:]
    generated = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    if not generated :
        raise RuntimeError("Reloaded model generated an empty response")

    print()
    print("expected:", expected)
    print("generated:", generated)

    try:
        parsed = json.loads(generated)
    except json.JSONDecodeError:
        print("JSON contract: FAIL")
    else :
        print("JSON contract: PASS")
        print("parsed:", parsed)

    print("G2 RELOAD + INFERENCE TECHNICAL PASS")

if __name__ == "__main__" :
    main()