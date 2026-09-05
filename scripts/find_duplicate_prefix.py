#!/usr/bin/env python3
"""筛选出 text 开头 N 个字符相同的条目，写入新文件。"""
import json
import sys


def main():
    src = "./train/chinese.jsonl"
    n = 15  # 开头取前 N 个字符
    out = "./train/chinese_duplicate_prefix.jsonl"

    if len(sys.argv) > 1:
        src = sys.argv[1]
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 3:
        out = sys.argv[3]

    from collections import defaultdict

    prefix_map = defaultdict(list)
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prefix = obj["text"][:n]
            prefix_map[prefix].append(obj)

    duplicate = [obj for group in prefix_map.values() if len(group) > 1 for obj in group]

    with open(out, "w", encoding="utf-8") as f:
        for obj in duplicate:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"重复前缀组数: {sum(1 for g in prefix_map.values() if len(g) > 1)}")
    print(f"重复条目总数: {len(duplicate)}")
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()