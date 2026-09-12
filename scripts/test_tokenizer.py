#!/usr/bin/env python3
"""分词器（tokenizer）测试脚本。

两种用法：
1) 内置样例快速自检（默认）：
       python scripts/test_tokenizer.py
2) 指定 tokenizer 目录 + 从 jsonl 真实语料采样做往返一致性测试：
       python scripts/test_tokenizer.py --dir /path/to/tokenizer \
           --jsonl train/chinese.jsonl train/english.jsonl train/article.jsonl \
           --sample 200

检查项：
1. 模型元信息（词表大小、特殊 token id）
2. 中英文 encode / decode 往返一致性
3. 特殊 token 的添除
4. batch + padding
5. 长文本长度截断
6. 从 jsonl 采样真实文本的往返一致性（数据分布下验证）
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = BASE_DIR / "tokenizer"

PASS = 0
FAIL = 0

# 内置样例（会受 training_tokenizer.py 的 NFKC 归一化影响）
BUILTIN_SAMPLES = [
    ("中文", "人工智能正在改变世界。"),
    ("英文", "The Earth orbits the Sun because"),
    ("中英混合", "AI 可以处理文本、图像和音频。artificial intelligence 2024"),
    ("标点符号", "Hello, world! （测试）：多符号——连接符；分号。"),
    ("数字", "圆周率约等于 3.1415926，而光速约为 299792458 m/s。"),
]


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))


def load_tokenizer(path: Path):
    if not (path / "tokenizer.json").exists():
        raise FileNotFoundError(f"未找到 tokenizer.json in {path}")
    return AutoTokenizer.from_pretrained(str(path))


def load_texts(paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for fp in paths:
        count = 0
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            texts.append(json.loads(line)["text"])
            count += 1
        print(f"  {fp.name}: {count} 条")
    return texts


def run_meta_and_builtin(tok) -> None:
    print("== 元信息 ==")
    print(f"  vocab_size = {tok.vocab_size}")
    print(f"  model_max_length = {tok.model_max_length}")
    print(f"  pad / unk / bos / eos id = "
          f"{tok.pad_token_id}/{tok.unk_token_id}/{tok.bos_token_id}/{tok.eos_token_id}")
    check("词表大小为正", bool(tok.vocab_size and tok.vocab_size > 0), str(tok.vocab_size))
    check("特殊 token id 均已定义", None not in
          (tok.pad_token_id, tok.unk_token_id, tok.bos_token_id, tok.eos_token_id))

    print("\n== 内置样例往返一致性 ==")
    for name, text in BUILTIN_SAMPLES:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        check(f"{name}: 往返一致", decoded == text, f"{len(ids)} tok | {decoded[:30]!r}")

    print("\n== 特殊 token ==")
    raw = "你好"
    ids_plain = tok.encode(raw, add_special_tokens=False)
    ids_special = tok.encode(raw, add_special_tokens=True)
    check("默认自动加 BOS/EOS", ids_special[0] == tok.bos_token_id
          and ids_special[-1] == tok.eos_token_id,
          f"{ids_plain} vs {ids_special}")
    check("skip_special_tokens 解码剥离特殊 token",
          tok.decode(ids_special, skip_special_tokens=True) == raw)

    print("\n== batch 与截断 ==")
    out = tok(["今天天气不错。", "GPT-2 is a model."], padding=True, truncation=True,
              return_tensors="pt")
    check("含 padding 的 batch 已生成", out["input_ids"].shape[0] == 2,
          str(tuple(out["input_ids"].shape)))
    long_text = "科学技术的进步正在重塑我们的日常生活。" * 200
    out_len = len(tok.encode(long_text, max_length=32, truncation=True))
    check("超过 max_length 被截断到 32", out_len == 32, f"实际 {out_len} tok")


def run_data_roundtrip(tok, texts: list[str], sample: int) -> None:
    print("\n== jsonl 数据往返一致性 ==")
    sample_texts = texts[:sample]
    ok = 0
    fails: list[tuple[str, str]] = []
    for t in sample_texts:
        ids = tok.encode(t, add_special_tokens=False)
        if tok.decode(ids, skip_special_tokens=True) == t:
            ok += 1
        elif len(fails) < 5:
            fails.append((t, tok.decode(ids, skip_special_tokens=True)))
    total = len(sample_texts)
    rate = ok / total * 100 if total else 100.0
    check(f"采样 {total} 条往返一致率应 >=90%", rate >= 90.0, f"{rate:.1f}% ({ok}/{total})")
    if fails:
        print("  不一致样例（多数因 NFKC 将全角标点折叠为半角导致）：")
        for orig, dec in fails:
            print(f"    原: {orig[:40]!r} => 解: {dec[:40]!r}")


def parse_args():
    p = argparse.ArgumentParser(description="分词器测试")
    p.add_argument("--dir", type=Path, default=TOKENIZER_DIR, help="tokenizer 目录")
    p.add_argument("--jsonl", type=Path, nargs="+", default=None,
                   help="从指定 jsonl 采样真实文本做往返测试")
    p.add_argument("--sample", type=int, default=200, help="每个全局抽样的条数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tok = load_tokenizer(args.dir)
    run_meta_and_builtin(tok)

    if args.jsonl:
        texts = load_texts(args.jsonl)
        run_data_roundtrip(tok, texts, args.sample)

    print("\n== 总结 ==")
    print(f"  PASS={PASS}  FAIL={FAIL}")
    if FAIL:
        print("  存在失败项")
        raise SystemExit(1)
    print("  全部通过")


if __name__ == "__main__":
    main()