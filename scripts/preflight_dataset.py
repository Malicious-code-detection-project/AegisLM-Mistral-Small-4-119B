from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict, cast

from transformers import AutoProcessor


class SplitSpec(TypedDict):
    path: Path
    count: int
    sha256: str


MODEL_ROOT = Path("model/mistral-small-4-119b-2603-bf16-local")

DATA_ROOT = Path("data/processed/phase-f-source-decision-v1")

EXPECTED_SPLITS: dict[str, SplitSpec] = {
    "train": {
        "path": DATA_ROOT / "train.jsonl",
        "count": 10_000,
        "sha256": ("2d46c7d8161cbf97cc4efeb1a8c223311f4260ccf7a333dbe3cd71b73715f322"),
    },
    "validation": {
        "path": DATA_ROOT / "validation.jsonl",
        "count": 1_000,
        "sha256": ("ab576d0332282244b68711f5fb7129c171cd0b95784f914cf5a7255e880d5481"),
    },
}

MANIFEST_PATH = DATA_ROOT / "dataset_manifest.json"
EXPECTED_MANIFEST_SHA256 = (
    "b987d174657061b0d0cdf263f738025514abbd9e3bb054251ca7d9d1777daa4e"
)

EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "9a72abf082ce57817ed0186719aeede75e51579c8131c95afd62b736485352ef"
)

EXPECTED_ROLES = ["system", "user", "assistant"]
ALLOWED_ASSESSMENTS = {
    "present",
    "not_observed",
    "uncertain",
}
FORBIDDEN_MARKERS = ("[THINK]", "[/THINK]")
MAX_SEQUENCE_LENGTH = 2_048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def to_input_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        raise TypeError(f"Unsupported tokenized output type: {type(value).__name__}")

    if value and isinstance(value[0], list):
        value = value[0]

    if not all(isinstance(token_id, int) for token_id in value):
        raise TypeError("Tokenized output contains non-integer token IDs")

    return cast(list[int], value)


def percentile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    idx = max(
        0,
        math.ceil(probability * len(ordered)) - 1,
    )

    return ordered[idx]


def validate_split(
    *,
    split: str,
    path: Path,
    expected_count: int,
    expected_sha256: str,
    tokenizer: Any,
    all_record_ids: set[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{split}: SHA-256 mismatch\n"
            f"expected={expected_sha256}\n"
            f"actual={actual_sha256}"
        )

    count = 0
    token_lengths: list[int] = []
    target_token_lengths: list[int] = []
    assessment_counts: Counter[str] = Counter()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"{split}:{line_number} : blank row")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{split}:{line_number}: invalid JSON") from exc

            if not isinstance(record, dict):
                raise TypeError(f"{split}:{line_number}: row is not an object")

            if set(record) != {"id", "messages"}:
                raise RuntimeError(f"{split}:{line_number}: invalid top-level keys")

            record_id = record["id"]
            if not isinstance(record_id, str) or not record_id:
                raise RuntimeError(f"{split}:{line_number}: invalid id")

            if record_id in all_record_ids:
                raise RuntimeError(f"{split}:{line_number}: duplicate id {record_id}")
            all_record_ids.add(record_id)

            messages_raw = record["messages"]

            if not isinstance(messages_raw, list):
                raise TypeError(f"{split}:{line_number}: messages not list")

            if not all(isinstance(message, dict) for message in messages_raw):
                raise TypeError(f"{split}:{line_number}: message is not an object")

            messages = cast(
                list[dict[str, Any]],
                messages_raw,
            )

            if len(messages) != 3:
                raise RuntimeError(f"{split}:{line_number}: message count != 3")

            roles = [message.get("role") for message in messages]

            if roles != EXPECTED_ROLES:
                raise RuntimeError(f"{split}:{line_number}: roles={roles}")

            for message in messages:
                if set(message) != {"role", "content"}:
                    raise RuntimeError(f"{split}:{line_number}: invalid message keys")

                content = message["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        f"{split}:{line_number}: invalid message content"
                    )

            system_prompt = messages[0]["content"]
            system_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

            if system_hash != EXPECTED_SYSTEM_PROMPT_SHA256:
                raise RuntimeError(
                    f"{split}:{line_number}: system prompt hash mismatch"
                )

            try:
                target = json.loads(messages[2]["content"])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{split}:{line_number}: assistant target is not JSON"
                ) from exc

            if (
                not isinstance(target, dict)
                or set(target) != {"assessment"}
                or target["assessment"] not in ALLOWED_ASSESSMENTS
            ):
                raise RuntimeError(
                    f"{split}:{line_number}: assistant target contract failed"
                )

            assessment_counts[target["assessment"]] += 1

            if any(
                marker in message["content"]
                for message in messages
                for marker in FORBIDDEN_MARKERS
            ):
                raise RuntimeError(f"{split}:{line_number}: raw THINK marker found")

            full_rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort="none",
            )

            if any(marker in full_rendered for marker in FORBIDDEN_MARKERS):
                raise RuntimeError(
                    f"{split}:{line_number}: rendered THINK marker found"
                )

            full_ids = to_input_ids(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    reasoning_effort="none",
                )
            )

            prompt_ids = to_input_ids(
                tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                    reasoning_effort="none",
                )
            )

            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(
                    f"{split}:{line_number}: prompt-target boundary mismatch"
                )

            target_ids = full_ids[len(prompt_ids) :]
            if not target_ids:
                raise RuntimeError(f"{split}:{line_number}: empty target")

            if len(full_ids) > MAX_SEQUENCE_LENGTH:
                raise RuntimeError(
                    f"{split}:{line_number}: "
                    f"{len(full_ids)} tokens exceeds "
                    f"{MAX_SEQUENCE_LENGTH}"
                )

            if full_ids[-1] != tokenizer.eos_token_id:
                raise RuntimeError(f"{split}:{line_number}: missing final EOS")

            token_lengths.append(len(full_ids))
            target_token_lengths.append(len(target_ids))
            count += 1

            if count % 1_000 == 0:
                print(
                    f"[{split}] checked {count:,} rows",
                    flush=True,
                )

    if count != expected_count:
        raise RuntimeError(f"{split}: count mismatch {count}!={expected_count}")

    return {
        "path": str(path),
        "sha256": actual_sha256,
        "count": count,
        "assessment_counts": dict(sorted(assessment_counts.items())),
        "token_length": {
            "min": min(token_lengths),
            "p50": percentile(token_lengths, 0.50),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths),
        },
        "assistant_target_token_length": {
            "min": min(target_token_lengths),
            "max": max(target_token_lengths),
        },
        "over_2048_count": 0,
        "think_marker_count": 0,
    }


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)

    manifest_sha256 = sha256_file(MANIFEST_PATH)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "dataset manifest SHA-256 mismatch\n"
            f"expected={EXPECTED_MANIFEST_SHA256}\n"
            f"actual={manifest_sha256}"
        )

    print("[load] local Mistral processor", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer

    all_record_ids: set[str] = set()
    reports = {}

    for split, spec in EXPECTED_SPLITS.items():
        print(f"[start] {split}", flush=True)
        reports[split] = validate_split(
            split=split,
            path=spec["path"],
            expected_count=spec["count"],
            expected_sha256=spec["sha256"],
            tokenizer=tokenizer,
            all_record_ids=all_record_ids,
        )

    result = {
        "model_root": str(MODEL_ROOT),
        "manifest_sha256": manifest_sha256,
        "system_prompt_sha256": (EXPECTED_SYSTEM_PROMPT_SHA256),
        "reasoning_effort": "none",
        "sequence_length": MAX_SEQUENCE_LENGTH,
        "unique_record_ids": len(all_record_ids),
        "splits": reports,
    }

    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()

    print("STEP 3-B PASS : all 11,000 dataset rows passed tokenizer preflight")


if __name__ == "__main__":
    main()
