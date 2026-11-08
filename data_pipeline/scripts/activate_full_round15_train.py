#!/usr/bin/env python3
"""Use all cleaned round15 records for training with the original valid/test sets."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data_pipeline" / "staging" / "round15"
DEST = ROOT / "data" / "train"
FILES = ("natural_zh.jsonl", "natural_en.jsonl", "article_zh.jsonl", "article_en.jsonl")
SPLITS = ("train", "valid", "test")


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text.casefold())


def main() -> None:
    outputs: dict[str, list[dict[str, object]]] = {}
    train_ids: set[str] = set()
    train_texts: set[str] = set()

    for name in FILES:
        merged = []
        for split in SPLITS:
            source = SOURCE / split / name
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != split:
                    raise ValueError(f"Unexpected split label in {source}: {record.get('id')}")
                record_id = record["id"]
                text = normalized(record["text"])
                if record_id in train_ids or text in train_texts:
                    raise ValueError(f"Duplicate record in round15: {record_id}")
                train_ids.add(record_id)
                train_texts.add(text)
                record["round15_split"] = split
                record["split"] = "train"
                merged.append(record)
        outputs[name] = merged

    for folder in (ROOT / "valid", ROOT / "test"):
        for source in folder.glob("*.jsonl"):
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("id") in train_ids or normalized(record["text"]) in train_texts:
                    raise ValueError(f"Training data overlaps {source}: {record.get('id')}")

    DEST.mkdir(parents=True, exist_ok=True)
    for name, records in outputs.items():
        target = DEST / name
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(target)
        print(f"{target}: {len(records)} records")
    print(f"Total training records: {len(train_ids)}")


if __name__ == "__main__":
    main()
