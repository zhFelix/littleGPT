#!/usr/bin/env python3
"""分别统计每个 jsonl 文件的文档数、字符数、token 数等指标。

用法：
  python scripts/count_tokens_per_file.py data/train/*.jsonl
  python scripts/count_tokens_per_file.py data/train/natural_zh.jsonl data/prepared/qa_zh.jsonl

默认读取 ./tokenizer，可通过 --tokenizer 指定。
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="jsonl 文件路径")
    parser.add_argument("--tokenizer", default="tokenizer", help="tokenizer 目录")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    header = f"{'文件':<38s}{'文档':>7s}{'字符':>10s}{'token':>10s}{'平均token/文档':>14s}"
    print(header)
    print("-" * len(header))

    for path in args.files:
        docs = chars = tokens = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            docs += 1
            chars += len(text)
            tokens += len(ids)
        avg = tokens / docs if docs else 0.0
        print(f"{str(Path(path)):<38s}{docs:7d}{chars:10d}{tokens:10d}{avg:14.1f}")


if __name__ == "__main__":
    main()