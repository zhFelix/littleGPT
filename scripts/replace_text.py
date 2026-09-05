#!/usr/bin/env python3
"""批量替换 chinese.jsonl 中 text 包含指定子句的条目。

输入替换文件格式（示例 replace.txt）：
    第一行：     要替换的子句（待替换文本）
    第二行起：   若干替换后的文本（按顺序循环使用）

用法：
    python3 replace_text.py <替换文件> [数据文件] [输出文件] [--in-place]
    默认数据: ./train/chinese.jsonl
    默认输出: ./train/chinese_replaced.jsonl
    --in-place 直接原地修改数据文件（内部先写临时文件再原子替换，安全）
"""
import json
import os
import sys


def main():
    args = sys.argv[1:]
    in_place = "--in-place" in args
    args = [a for a in args if a != "--in-place"]

    replace_file = args[0] if len(args) > 0 else "./replace.txt"
    src = os.path.abspath(args[1] if len(args) > 1 else "./train/chinese.jsonl")
    out = src if in_place else (os.path.abspath(args[2] if len(args) > 2 else "./train/chinese_replaced.jsonl"))

    if src == out and not in_place:
        print("错误：输出文件不能与输入数据文件相同，否则会先截断原文件。"
              "\n请改用 --in-place 原地修改，或用不同的输出文件。")
        sys.exit(1)

    with open(replace_file, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if len(lines) < 2:
        print("替换文件至少需要一行待替换文本和一行替换文本。")
        sys.exit(1)

    old_text = lines[0]
    replacements = lines[1:]
    rep_count = len(replacements)

    matched = 0
    replaced = 0
    idx = 0
    tmp = out + ".tmp"
    # 先写临时文件，全部成功后再原子替换为输出文件，避免写一半损坏
    with open(src, "r", encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"]
            if old_text in text:
                matched += 1
                text = text.replace(old_text, replacements[idx % rep_count])
                idx += 1
                replaced += 1
                obj["text"] = text
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, out)

    print(f"待替换子句: {old_text!r}")
    print(f"替换文本: {len(replacements)} 条，循环使用")
    print(f"匹配条目数: {matched}")
    print(f"实际替换次数: {replaced}")
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()