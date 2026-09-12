#!/usr/bin/env python3
"""生成可训练的 QA jsonl 语料。

输入：包含 question / answer（可选带 language / domain / topic / id）的 jsonl；
输出：训练可直接消费的 jsonl，自动补齐 text 字段（question+answer 拼接），
必要时补 id。既有训练加载器读取 text，因此无需改训练代码。

用法：
    python scripts/build_qa_corpus.py 源.jsonl --output train/qa.jsonl [--append]
    python scripts/build_qa_corpus.py 源1.jsonl 源2.jsonl --output train/qa.jsonl

拼接模板（按 language）：
    zh -> "问：{question}\\n答：{answer}"
    其它 -> "Q: {question}\\nA: {answer}"
"""
import argparse
import json
from pathlib import Path


def build_text(q: str, a: str, lang: str) -> str:
    if lang == "zh":
        return f"问：{q}\n答：{a}"
    return f"Q: {q}\nA: {a}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs="+", type=Path, help="源 jsonl（含 question/answer）")
    parser.add_argument("--output", type=Path, required=True, help="输出 jsonl")
    parser.add_argument("--append", action="store_true", help="追加到已存在的输出文件")
    args = parser.parse_args()

    existing_texts: set[str] = set()
    if args.append and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_texts.add(json.loads(line)["text"])

    added = 0
    skipped = 0
    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as fo:
        for src in args.src:
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                q = r.get("question", "").strip()
                a = r.get("answer", "").strip()
                if not q or not a:
                    skipped += 1
                    continue
                lang = r.get("language", "zh")
                text = r.get("text") or build_text(q, a, lang)
                if text in existing_texts:
                    skipped += 1
                    continue
                lang = lang[:2].lower()
                rid = r.get("id") or f"qa-{lang}-{len(existing_texts)+added+1:04d}"
                out = {
                    "id": rid,
                    "split": r.get("split", "train"),
                    "language": lang,
                    "domain": r.get("domain", "general"),
                    "topic": r.get("topic", "general"),
                    "question": q,
                    "answer": a,
                    "text": text,
                }
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")
                existing_texts.add(text)
                added += 1

    print(f"新增 {added} 条，跳过（缺字段/重复）{skipped} 条，共写入 {args.output}")


if __name__ == "__main__":
    main()