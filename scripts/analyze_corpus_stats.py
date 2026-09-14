#!/usr/bin/env python3
"""统计 corpus 的 token 指标：文档数、总字符、总 token、独词 token 数、均值、分位数、最大等。

用法：
  python scripts/analyze_corpus_stats.py data/train/article_zh.jsonl [data/train/natural_zh.jsonl ...]
"""
import argparse
import json
import sys

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="jsonl 文件路径")
    parser.add_argument("--tokenizer", default="tokenizer")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    docs, total_chars, total_tokens = 0, 0, 0
    unique = set()

    for path in args.files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                docs += 1
                total_chars += len(text)
                total_tokens += len(ids)
                unique.update(ids)
        print(f"{path}: ok")

    print("=" * 40)
    print(f"文档数       {docs}")
    print(f"总字符       {total_chars}")
    print(f"总 token     {total_tokens}")
    print(f"unique tokens {len(unique)}")
    if docs:
        print(f"平均 token/文档 {total_tokens / docs:.1f}")
        print(f"平均字符/文档   {total_chars / docs:.1f}")


if __name__ == "__main__":
    main()