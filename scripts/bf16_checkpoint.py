import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open

root = Path("model/mistral-small-4-119b-2603-bf16-local")

required = [
    "config.json",
    "model.safetensors.index.json",
    "conversion_manifest.json",
]

for name in required:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"필수 파일 누락: {path}")

config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
manifest = json.loads((root / "conversion_manifest.json").read_text())

weight_map = index["weight_map"]
shards = sorted(set(weight_map.values()))
missing_shards = [shard for shard in shards if not (root / shard).is_file()]

actual_keys = set()
dtype_counts: Counter[str] = Counter()
fp8_keys = []
scale_keys = []

scale_suffixes = (
    "weight_scale_inv",
    "gate_up_proj_scale_inv",
    "down_proj_scale_inv",
    "up_proj_scale_inv",
    "activation_scale",
    "gate_up_proj_activation_scale",
    "down_proj_activation_scale",
)

for shard_name in shards:
    with safe_open(
        root / shard_name,
        framework="pt",
        device="cpu",
    ) as handle:
        # safe_open은 일반 dict가 아니며 공식 API 가 keys() 순회를 사용한다.
        for key in handle.keys():  # noqa: SIM118
            actual_keys.add(key)

            dtype = str(handle.get_slice(key).get_dtype())
            dtype_counts[dtype] += 1

            upper_dtype = dtype.upper()
            if "FLOAT8" in upper_dtype or "F8_" in upper_dtype:
                fp8_keys.append(key)

            if key.endswith(scale_suffixes):
                scale_keys.append(key)

print("model directory:", root)
print("dtype:", config.get("dtype"))
print(
    "quantization_config:",
    config.get("quantization_config"),
)
print("index tensor count:", len(weight_map))
print("actual tensor count:", len(actual_keys))
print("shard count:", len(shards))
print("missing shard count:", len(missing_shards))
print("stored bytes:", index["metadata"]["total_size"])
print("dtype counts:", dict(dtype_counts))
print("remaining FP8 tensors:", len(fp8_keys))
print("remaining scale tensors:", len(scale_keys))
print("source revision:", manifest["source_revision"])
print(
    "converted tensor count:",
    manifest["converted_tensor_count"],
)

assert config.get("dtype") == "bfloat16"
assert "quantization_config" not in config
assert "quantization_config" not in config.get(
    "text_config",
    {},
)
assert not missing_shards
assert set(weight_map) == actual_keys
assert not fp8_keys
assert not scale_keys

print("STEP 1 PASS: BF16 checkpoint structure verified")
