from pathlib import Path

from transformers import AutoConfig, AutoProcessor

root = Path("model/mistral-small-4-119b-2603-bf16-local")

print("[1/3] Loading config locally")
config = AutoConfig.from_pretrained(
    root,
    local_files_only=True,
    trust_remote_code=False,
)

print("[2/3] Loading processor locally")
processor = AutoProcessor.from_pretrained(
    root,
    local_files_only=True,
    trust_remote_code=False,
)

tokenizer = getattr(processor, "tokenizer", None)
if tokenizer is None:
    raise RuntimeError("Processor에서 tokenizer를 찾을 수 없습니다.")

if not getattr(tokenizer, "chat_template", None):
    raise RuntimeError("Tokenizer에 chat_template가 없습니다.")

messages = [
    {
        "role": "system",
        "content": ("You are a careful security analysis assistant."),
    },
    {
        "role": "user",
        "content": "Reply with exactly: OK",
    },
]

print("[3/3] Rendering chat template")
rendered = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    reasoning_effort="none",
)

tokenized = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="none",
)

if isinstance(tokenized, dict):
    input_ids = tokenized["input_ids"]
else:
    input_ids = tokenized

if hasattr(input_ids, "shape"):
    token_count = input_ids.shape[-1]
elif input_ids and isinstance(input_ids[0], list):
    token_count = len(input_ids[0])
else:
    token_count = len(input_ids)

think_count = rendered.count("[THINK]") + rendered.count("[/THINK]")

print()
print("config class:", type(config).__name__)
print("model type:", config.model_type)
print("config dtype:", config.dtype)
print(
    "quantization config:",
    getattr(config, "quantization_config", None),
)
print("processor class:", type(processor).__name__)
print("tokenizer class:", type(tokenizer).__name__)
print("vocab size:", len(tokenizer))
print("BOS token:", repr(tokenizer.bos_token))
print("EOS token:", repr(tokenizer.eos_token))
print("PAD token:", repr(tokenizer.pad_token))
print("rendered token count:", token_count)
print("THINK marker count:", think_count)
print("rendered preview:", repr(rendered[:500]))

assert config.dtype is not None
assert getattr(config, "quantization_config", None) is None
assert token_count > 0
assert think_count == 0

print()
print("STEP 2 PASS: local config, processor and chat template verified")
