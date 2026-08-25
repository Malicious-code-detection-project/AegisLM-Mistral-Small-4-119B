import json
from pathlib import Path

from transformers import AutoProcessor

model_root = Path("model/mistral-small-4-119b-2603-bf16-local")

dataset_paths = [
    Path("data/processed/phase-f-source-decision-v1/train.jsonl"),
    Path("data/processed/phase-f-source-decision-v1/validation.jsonl"),
]

processor = AutoProcessor.from_pretrained(
    model_root,
    local_files_only=True,
    trust_remote_code=False,
)
tokenizer = processor.tokenizer


def to_input_ids(value):
    if isinstance(value, dict):
        value = value["input_ids"]

    if hasattr(value, "tolist"):
        value = value.tolist()

    if value and isinstance(value[0], list):
        value = value[0]

    return list(value)


def load_first_record(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                return line_number, json.loads(line)

    raise RuntimeError(f"빈 dataset 파일: {path}")


for path in dataset_paths:
    if not path.is_file():
        raise FileNotFoundError(path)

    line_number, record = load_first_record(path)

    if set(record) != {"id", "messages"}:
        raise RuntimeError(f"{path}: top-level keys 오류: {sorted(record)}")

    messages = record["messages"]
    roles = [message.get("role") for message in messages]

    if roles != ["system", "user", "assistant"]:
        raise RuntimeError(f"{path}: role 순서 오류: {roles}")

    for message in messages:
        if set(message) != {"role", "content"}:
            raise RuntimeError(f"{path}: message keys 오류: {sorted(message)}")
        if not isinstance(message["content"], str):
            raise TypeError(f"{path}: content가 문자열이 아닙니다.")
        if not message["content"].strip():
            raise RuntimeError(f"{path}: 빈 content가 있습니다.")

    raw_marker_count = sum(
        message["content"].count("[THINK]") + message["content"].count("[/THINK]")
        for message in messages
    )

    # 학습용 전체 sequence: assistant 정답까지 포함한다.
    full_rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        reasoning_effort="none",
    )
    full_ids = to_input_ids(
        processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            reasoning_effort="none",
        )
    )

    # 추론용 prompt prefix: assistant 정답 직전까지다.
    prompt_messages = messages[:-1]
    prompt_ids = to_input_ids(
        processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort="none",
        )
    )

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"{path}: prompt와 assistant target 경계 불일치")

    assistant_target_ids = full_ids[len(prompt_ids) :]
    rendered_marker_count = full_rendered.count("[THINK]") + full_rendered.count(
        "[/THINK]"
    )

    assistant_text = messages[-1]["content"]

    print()
    print("file:", path)
    print("line:", line_number)
    print("record id:", record["id"])
    print("roles:", roles)
    print(
        "content character counts:",
        [len(message["content"]) for message in messages],
    )
    print("prompt token count:", len(prompt_ids))
    print(
        "assistant target token count:",
        len(assistant_target_ids),
    )
    print("full token count:", len(full_ids))
    print("raw THINK markers:", raw_marker_count)
    print(
        "rendered THINK markers:",
        rendered_marker_count,
    )
    print(
        "assistant content found in rendered text:",
        assistant_text in full_rendered,
    )
    print(
        "ends with EOS:",
        full_ids[-1] == tokenizer.eos_token_id,
    )

    assert assistant_target_ids
    assert len(full_ids) <= 2048
    assert raw_marker_count == 0
    assert rendered_marker_count == 0
    assert assistant_text in full_rendered
    assert full_ids[-1] == tokenizer.eos_token_id

print()
print("STEP 3-A PASS: train and validation samples match the Mistral training template")
