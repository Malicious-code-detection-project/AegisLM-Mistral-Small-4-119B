import json
from pathlib import Path

root = Path("model/mistral-small-4-119b-2603-bf16-local")
path = root / "model.safetensors.index.json"
index = json.loads(path.read_text(encoding="utf-8"))

keys = sorted(
    key
    for key in index["weight_map"]
    if ".layers.0." in key
    and ("proj" in key or ".experts." in key or ".gate." in key)
)

print("index: ", path)
print("matched keys:", len(keys))
for key in keys:
    print(key)
