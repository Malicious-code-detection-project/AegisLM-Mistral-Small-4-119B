import json
from math import prod
from pathlib import Path

from safetensors import safe_open

root = Path("model/mistral-small-4-119b-2603-bf16-local")
path = root / "model.safetensors.index.json"
index = json.loads(path.read_text(encoding="utf-8"))

keys = [
    "language_model.model.layers.0.mlp.experts.gate_up_proj",
    "language_model.model.layers.0.mlp.experts.down_proj",
]

for key in keys :
    shard = root / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle :
        shape = handle.get_slice(key).get_shape()

    print(key)
    print("  shape:", shape)
    print("  numel:", prod(shape))
    print("  exceeds INT_MAX:", prod(shape) > 2_147_483_647)