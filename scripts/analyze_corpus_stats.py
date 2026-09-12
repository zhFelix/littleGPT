#!/usr/bin/env python3
"""用 tokenizer 对 jsonl 训练语料做语料统计。

用法：
    python scripts/analyze_corpus_stats.py [--dir tokenizer] train/chinese.jsonl train/english.jsonl train/article.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = BASE_DIR / "tokenizer"

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z]+")


def cjk_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def classify(decoded: str):
    """按 decode 还原后的子串归类：zh / en / other。"""
    if CJK_RE.search(decoded):
        return "zh"
    if WORD_RE.search(decoded):
        return "en"
    return "other"


def analyze(tok, texts) -> dict:
    n_doc = len(texts)
    n_char = sum(len(t) for t in texts)
    n_han = sum(cjk_count(t) for t in texts)
    n_word = sum(len(WORD_RE.findall(t)) for t in texts)

    per_plain: list[int] = []   # 无特殊 token
    per_special: list[int] = [] # 含 bos/eos
    stats = {"zh_tok": 0, "en_tok": 0, "other_tok": 0, "eos": 0, "unk": 0}

    for t in texts:
        p = tok.encode(t, add_special_tokens=False)
        s = tok.encode(t, add_special_tokens=True)
        per_plain.append(len(p))
        per_special.append(len(s))

        stats["eos"] += sum(1 for i in s if i == tok.eos_token_id)
        stats["unk"] += sum(1 for i in p if i == tok.unk_token_id)

        for i in p:
            if i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id, None):
                continue
            sub = tok.decode([i]).strip()  # ByteLevel 需 decode 还原真实字符再分类
            if sub:
                stats[classify(sub) + "_tok"] += 1

    per_special_sorted = sorted(per_special)
    def pctile(arr, q):
        if not arr:
            return 0
        k = max(0, min(len(arr) - 1, int((q / 100) * len(arr))))
        return arr[k]

    total_special = sum(stats["zh_tok"] + stats["en_tok"] + stats["other_tok"]
                        for _ in [0] if False)  # placeholder (recomputed below)
    return {
        "n_doc": n_doc,
        "n_char": n_char,
        "n_han": n_han,
        "n_word": n_word,
        "total_tok_plain": sum(per_plain),
        "total_tok_special": sum(per_special),
        "avg_tok": (sum(per_special) / n_doc) if n_doc else 0,
        "p50": pctile(per_special_sorted, 50),
        "p90": pctile(per_special_sorted, 90),
        "max_tok": max(per_special) if per_special else 0,
        "zh_tok": stats["zh_tok"],
        "en_tok": stats["en_tok"],
        "other_tok": stats["other_tok"],
        "eos": stats["eos"],
        "unk": stats["unk"],
    }


def fmt_pct(x):
    return f"{x*100:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path, help="jsonl 语料文件")
    parser.add_argument("--dir", type=Path, default=TOKENIZER_DIR, help="tokenizer 目录")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.dir))

    hdr = ("文件", "文档", "字符", "token", "均token/文档", "P50", "P90", "最大",
           "tokens/汉字", "tokens/word", "中Token%", "英Token%", "EOS%", "UNK")
    print("{:<22}{:>7}{:>10}{:>9}{:>12}{:>7}{:>7}{:>7}{:>10}{:>10}{:>9}{:>9}{:>7}{:>8}".format(*hdr))
    print("-" * 130)

    agg = {k: 0 for k in ["n_doc", "n_char", "n_word", "total_tok_plain", "total_tok_special",
                          "zh_tok", "en_tok", "other_tok", "eos", "unk"]}
    for f in args.files:
        texts = [json.loads(l)["text"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        r = analyze(tok, texts)
        for k in agg:
            agg[k] += r[k]
        zh_pct = r["zh_tok"] / (r["zh_tok"] + r["en_tok"] + r["other_tok"])
        en_pct = r["en_tok"] / (r["zh_tok"] + r["en_tok"] + r["other_tok"])
        eos_pct = r["eos"] / r["total_tok_special"]
        tk_per_han = (r["total_tok_plain"] / r["n_han"]) if r["n_han"] else float("nan")
        tk_per_word = (r["total_tok_plain"] / r["n_word"]) if r["n_word"] else float("nan")
        print("{:<22}{:>7}{:>10}{:>9}{:>12}{:>7}{:>7}{:>7}{:>10}{:>10}{:>9}{:>9}{:>7}{:>8}".format(
            f.name, r["n_doc"], r["n_char"], r["total_tok_plain"], f"{r['avg_tok']:.1f}",
            r["p50"], r["p90"], r["max_tok"], f"{tk_per_han:.2f}", f"{tk_per_word:.2f}",
            fmt_pct(zh_pct), fmt_pct(en_pct), fmt_pct(eos_pct), r["unk"]))

    # 汇总行
    t_all = agg["zh_tok"] + agg["en_tok"] + agg["other_tok"]
    t_han = sum(cjk_count(t) for f in args.files for t in
                [json.loads(l)["text"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])
    t_words = sum(len(WORD_RE.findall(t)) for f in args.files for t in
                  [json.loads(l)["text"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()])
    zh_pct = agg["zh_tok"] / t_all
    en_pct = agg["en_tok"] / t_all
    eos_pct = agg["eos"] / agg["total_tok_special"]
    print("-" * 130)
    print("汇总".ljust(18),
          f"{agg['n_doc']:>7}{agg['n_char']:>10}{agg['total_tok_plain']:>9}  (总token含特殊={agg['total_tok_special']:,})".ljust(19),
          f"{fmt_pct(zh_pct)}".rjust(9), f"{fmt_pct(en_pct)}".rjust(9),
          f"{fmt_pct(eos_pct)}".rjust(7), f"{agg['unk']:>8}")
    print(f"\ntokens/汉字(汇总) = {agg['total_tok_plain']/t_han:.2f}   tokens/word(汇总) = {agg['total_tok_plain']/t_words:.2f}")


if __name__ == "__main__":
    main()