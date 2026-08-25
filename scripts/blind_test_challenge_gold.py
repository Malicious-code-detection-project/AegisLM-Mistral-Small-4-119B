import json
from pathlib import Path

paths = [
    Path(
        "data/processed/phase-f-source-fresh-blind-500-v1/"
        "challenge.jsonl"
    ),
    Path(
        "data/processed/"
        "phase-f-source-fresh-blind-contracts-500-v1/"
        "decision/challenge.jsonl"
    ),
]

for path in paths:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    first = rows[0]

    print()
    print("path:", path)
    print("count:", len(rows))
    print("unique ids:", len({str(row["id"]) for row in rows}))
    print("top-level keys:", sorted(first))

    messages = first.get("messages")
    if isinstance(messages, list):
        print("roles:", [message.get("role") for message in messages])
        print("message count:", len(messages))