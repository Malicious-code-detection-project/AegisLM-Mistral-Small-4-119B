from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open
from safetensors.torch import load_file

SOURCE_REPO = "mistralai/Mistral-Small-4-119B-2603"
OFFICIAL_METHOD = (
    "https://huggingface.co/mistralai/"
    "Mistral-Small-4-119B-2603/blob/"
    "3b76d234c932e78dc989731bfc4c3b12c0a87918/README.md"
)

SCALE_SUFFIXES = (
    "weight_scale_inv",
    "gate_up_proj_scale_inv",
    "down_proj_scale_inv",
    "up_proj_scale_inv",
)

ACTIVATION_SCALE_SUFFIXES = (
    "activation_scale",
    "gate_up_proj_activation_scale",
    "down_proj_activation_scale",
)

FP8_DTYPES = tuple(
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, name, None)) is not None
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official Mistral FP8 weights to a local BF16 checkpoint"
    )

    parser.add_argument(
        "--source", type=Path, default=Path("model/mistral-small-4-119b-2603")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/mistral-small-4-119b-2603-bf16-local"),
    )

    parser.add_argument(
        "--revision",
        default=os.environ.get("MISTRAL_OFFICIAL_REVISION", ""),
    )

    parser.add_argument("--max-shard-size", default="10GB")
    return parser.parse_args()


def load_checkpoint(
    source: Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")

    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Invalid or empty weight_map")

    expected_shards = set(weight_map.values())
    actual_shards = {path.name for path in source.glob("model*.safetensors")}

    missing = expected_shards - actual_shards
    unexpected = actual_shards - expected_shards

    if missing:
        raise RuntimeError(f"Missing source shards: {sorted(missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected source shards: {sorted(unexpected)}")

    state_dict: dict[str, torch.Tensor] = {}

    for number, shard_name in enumerate(sorted(expected_shards), start=1):
        print(
            f"[load {number}/{len(expected_shards)}] {shard_name}",
            flush=True,
        )
        shard = load_file(str(source / shard_name), device="cpu")

        duplicate_keys = state_dict.keys() & shard.keys()
        if duplicate_keys:
            raise RuntimeError(f"Duplicate keys found: {sorted(duplicate_keys)[:5]}")

        state_dict.update(shard)

    expected_keys = set(weight_map)
    loaded_keys = set(state_dict)

    if loaded_keys != expected_keys:
        missing_keys = expected_keys - loaded_keys
        unexpected_keys = loaded_keys - expected_keys

        raise RuntimeError(
            "Index/state mismatch: "
            f"missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}"
        )

    return state_dict, index


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, int]:
    scale_keys = {key for key in state_dict if has_suffix(key, SCALE_SUFFIXES)}

    activation_scale_keys = {
        key for key in state_dict if has_suffix(key, ACTIVATION_SCALE_SUFFIXES)
    }

    initial_fp8 = sum(tensor.dtype in FP8_DTYPES for tensor in state_dict.values())

    if initial_fp8 == 0:
        raise RuntimeError(
            "No FP8 tensors found. Refusing to convert a non-FP8 checkpoint."
        )

    used_scale_keys: set[str] = set()
    converted = 0

    for key in list(state_dict):
        if key in scale_keys or key in activation_scale_keys:
            continue

        scale_key = scale_key_for(key, state_dict)
        if scale_key is None:
            continue

        state_dict[key] = descale_fp8_to_bf16(
            state_dict[key],
            state_dict[scale_key],
        )
        used_scale_keys.add(scale_key)
        converted += 1

        if converted % 100 == 0:
            print(f"[convert] {converted} tensors", flush=True)

    unused_scale_keys = scale_keys - used_scale_keys
    if unused_scale_keys:
        sample = sorted(unused_scale_keys)[:10]
        raise RuntimeError(
            f"Unused inverse-scale tensors: {len(unused_scale_keys)}; sample={sample}"
        )

    for key in scale_keys | activation_scale_keys:
        del state_dict[key]

    remaining_scale_keys = {
        key
        for key in state_dict
        if has_suffix(
            key,
            SCALE_SUFFIXES + ACTIVATION_SCALE_SUFFIXES,
        )
    }

    if remaining_scale_keys:
        raise RuntimeError(f"Scale tensors remain: {sorted(remaining_scale_keys)[:10]}")

    remaining_fp8 = [
        key for key, tensor in state_dict.items() if tensor.dtype in FP8_DTYPES
    ]

    if remaining_fp8:
        raise RuntimeError(f"FP8 tensors remain: {remaining_fp8[:10]}")

    dtype_counts = Counter(str(tensor.dtype) for tensor in state_dict.values())
    print(f"[convert] dtype counts : {dict(dtype_counts)}", flush=True)

    return {
        "input_fp8_tensor_count": initial_fp8,
        "converted_tensor_count": converted,
        "removed_inverse_scale_count": len(scale_keys),
        "removed_activation_scale_count": len(activation_scale_keys),
        "output_tensor_count": len(state_dict),
    }


def has_suffix(key: str, suffixes: tuple[str, ...]) -> bool:
    return any(key.endswith(suffix) for suffix in suffixes)


def scale_key_for(key: str, state_dict: dict[str, torch.Tensor]) -> str | None:
    if key.endswith(".weight"):
        candidate = key.rsplit(".weight", 1)[0] + ".weight_scale_inv"
        if candidate in state_dict:
            return candidate

    for suffix in SCALE_SUFFIXES[1:]:
        projection = suffix.removesuffix("_scale_inv")
        if key.endswith(f".{projection}"):
            candidate = key + "_scale_inv"
            if candidate in state_dict:
                return candidate

    return None


def descale_fp8_to_bf16(
    tensor: torch.Tensor,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    return (tensor.to(torch.bfloat16) * scale_inv.to(torch.bfloat16)).to(torch.bfloat16)


def copy_metadata(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        if not item.is_file():
            continue

        if item.name == "model.safetensors.index.json":
            continue

        if item.name.startswith("model") and item.suffix == ".safetensors":
            continue

        if item.name.startswith("consolidated") and item.suffix == ".safetensors":
            continue

        shutil.copy2(item, destination / item.name)


def rewrite_configs(destination: Path) -> None:
    config_path = destination / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.pop("quantization_config", None)
        text_config["dtype"] = "bfloat16"

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    params_path = destination / "params.json"
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))
        params.pop("quantization", None)
        params_path.write_text(
            json.dumps(params, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def verify_saved_checkpoint(destination: Path) -> dict[str, int]:
    index_path = destination / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("Output index was not generated")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    expected_keys = set(weight_map)

    actual_keys: set[str] = set()
    dtype_counts: Counter[str] = Counter()

    for shard_name in sorted(set(weight_map.values())):
        shard_path = destination / shard_name

        if not shard_path.is_file():
            raise RuntimeError(f"Missing output shard: {shard_path}")

        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as handle:
            # safe_open은 일반 dict가 아니며 공식 API가 keys() 순회를 사용한다.
            for key in handle.keys():  # noqa: SIM118
                if key in actual_keys:
                    raise RuntimeError(f"Duplicate output key: {key}")

                actual_keys.add(key)
                dtype = str(handle.get_slice(key).get_dtype())
                dtype_counts[dtype] += 1

                upper_dtype = dtype.upper()

                if "FLOAT8" in upper_dtype or "F8_" in upper_dtype:
                    raise RuntimeError(f"FP8 output tensor found: {key} ({dtype})")

                if has_suffix(
                    key,
                    SCALE_SUFFIXES + ACTIVATION_SCALE_SUFFIXES,
                ):
                    raise RuntimeError(f"Scale tensor found in output: {key}")

    if actual_keys != expected_keys:
        raise RuntimeError(
            "Output index mismatch: "
            f"index={len(expected_keys)}, actual={len(actual_keys)}"
        )

    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))

    if "quantization_config" in config:
        raise RuntimeError("Top-level quantization_config remains")

    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "quantization_config" in text_config:
        raise RuntimeError("text_config.quantization_config remains")

    print(f"[verify] dtype counts: {dict(dtype_counts)}", flush=True)

    return {
        "saved_tensor_count": len(actual_keys),
        "saved_shard_count": len(set(weight_map.values())),
        "saved_total_size": index.get("metadata", {}).get(
            "total_size",
            0,
        ),
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")

    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise RuntimeError(
            "--revision or MISTRAL_OFFICIAL_REVISIONmust be a 40-character SHA"
        )

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")

    if incomplete.exists():
        raise FileExistsError(
            f"Incomplete output already exists. Inspect it before removal{incomplete}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir()

    print(f"[source] {source}", flush=True)
    print(f"[output] {output}", flush=True)
    print(f"[revision] {args.revision}", flush=True)

    state_dict, source_index = load_checkpoint(source)
    conversion_stats = convert_state_dict(state_dict)

    print("[save] writing BF16 safetensors shards", flush=True)
    save_torch_state_dict(
        state_dict=state_dict,
        save_directory=incomplete,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=args.max_shard_size,
        metadata={
            "format": "pt",
            "source_repo": SOURCE_REPO,
            "source_revision": args.revision,
            "conversion": "official-fp8-descale-to-bf16",
        },
        safe_serialization=True,
        force_contiguous=True,
    )

    del state_dict
    gc.collect()

    copy_metadata(source, incomplete)
    rewrite_configs(incomplete)
    saved_stats = verify_saved_checkpoint(incomplete)

    manifest = {
        "schema_version": "aegislm.mistral-fp8-to-bf16.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_repo": SOURCE_REPO,
        "source_revision": args.revision,
        "source_directory": str(source),
        "source_stored_bytes": source_index.get(
            "metadata",
            {},
        ).get("total_size"),
        "output_directory": str(output),
        "output_dtype": "bfloat16",
        "max_shard_size": args.max_shard_size,
        "conversion_method": OFFICIAL_METHOD,
        **conversion_stats,
        **saved_stats,
    }

    (incomplete / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    incomplete.rename(output)

    print()
    print("BF16 conversion PASS")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
