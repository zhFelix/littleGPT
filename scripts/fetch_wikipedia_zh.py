#!/usr/bin/env python3
"""从 zh.wikipedia.org 抓取 A 级中文文章种子，规范化标题 + 重定向解析，
清洗正文（参考文献/參閱/外部連結章节、{{...}} 模板、公式残片、引文编号、
噪音空行）后按自然句分块，追加到 data/train/article_zh.jsonl。

用法：
  python scripts/fetch_wikipedia_zh.py                      # 全量抓 A 级
  python scripts/fetch_wikipedia_zh.py --dry --seed 牛顿运动定律   # 单条验证清洗
  python scripts/fetch_wikipedia_zh.py --priority A
"""
import argparse
import json
import random
import re
import sys
import time

import requests

API = "https://zh.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "littleGPT training corpus fetcher/1.0 (edu research; contact: local)"
}
SEED_FILE = "zh_wikipedia_article_seed_200.jsonl"
OUT_FILE = "data/train/article_zh.jsonl"

# 切块参数（参照现有 article_en，目标约 230 字符）
TARGET, MIN_LEN, MAX_LEN = 230, 140, 380

# 输出即丢弃的章节名（尾随引用/导航类）
DROPPED_SECTIONS = {
    "參閱", "參見", "参见", "参阅", "註釋", "注释", "腳註", "脚注",
    "參考文獻", "参考文献", "参考資料", "参考资料", "外部連結", "外部链接",
    "延伸閱讀", "延伸阅读", "相關條目", "相关条目", "外部鏈接",
    "注釋", "註記", "注記", "引用", "参考文献来源", "延伸阅读",
    "參考文獻及註釋", "註", "注释与参考",
}

# 公式/引文/导航等噪声正则
EQUATION_RE = re.compile(r"\s*\{\\displaystyle.*?\}\s*", re.S)        # {\displaystyle ...}
CITATION_RE = re.compile(r"[\[（(]\d[\d\s.,，-]*[）\])]")              # [1] (1) 等引文编号
ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")                             # &nbsp; 等
BRACKET_LINK_RE = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")          # 链接残留
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|文件|Image|Category):[^\]]*\]\]", re.I)


def api(params: dict) -> dict:
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    for attempt in range(8):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=40)
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            # 指数退避 + 抖动，规避连续请求限流
            delay = 1 + attempt * attempt
            time.sleep(delay + random.uniform(0, 0.5))
    raise RuntimeError(f"API request failed: {params.get('titles')}")


def resolve_title(title_seed: str) -> str | None:
    """先 query 解析重定向/规范化；missing 则用 search 兜底。返回合法标题或 None。"""
    d = api({
        "action": "query", "titles": title_seed, "redirects": "1", "prop": "info",
    })
    q = d.get("query", {})
    for redirect in q.get("redirects", []):
        title_seed = redirect.get("to", title_seed)
    for p in q.get("pages", []):
        if not p.get("missing"):
            return p["title"]
    # missing -> search 兜底
    s = api({
        "action": "query", "list": "search", "srsearch": title_seed,
        "srnamespace": "0", "srlimit": "3",
    })
    for hit in s.get("query", {}).get("search", []):
        title = hit["title"]
        if not any(x in title for x in ("消歧义", "消歧義", "列表")):
            return title
    return None


def fetch_extract(title: str) -> str:
    d = api({
        "action": "query", "titles": title, "redirects": "1",
        "prop": "extracts", "explaintext": "1",
    })
    for p in d.get("query", {}).get("pages", []):
        return p.get("extract", "") or ""
    return ""


def drop_trailing_sections(text: str) -> str:
    """按 \n== 拆分，丢弃被列入 DROPPED_SECTIONS 的章节及其后续内容。返回保留正文。"""
    parts = re.split(r"(?=\n=+ +\S)", text)
    kept = []
    for part in parts:
        m = re.match(r"\n=+ ?([^=]+?) *=*\n", part)
        if m and m.group(1).strip() in DROPPED_SECTIONS:
            break
        kept.append(part)
    return "".join(kept)


def clean_text(text: str) -> str:
    text = drop_trailing_sections(text)
    # 顶层说明性模板残留（如“本条目中，向量與标量分别用粗體...”）
    text = re.sub(r"^\s*本條目[^\n]*。\s*", "", text)
    text = re.sub(r"^\s*本条[^\n]*。\s*", "", text)
    text = FILE_LINK_RE.sub(" ", text)
    text = EQUATION_RE.sub(" ", text)
    text = BRACKET_LINK_RE.sub(r"\2", text)  # 保留链接可见文本
    text = citation_re_clean(text)
    text = ENTITY_RE.sub(" ", text)
    # 章节标题行（== xxx ==）与内层子标题（=== xxx ===）删除
    text = re.sub(r"\n=+ ?[^=\n]+? *=*\n", "\n", text)
    # 折叠空行与多余空白
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def citation_re_clean(text: str) -> str:
    # 括号内引文 (英文人名+数字) 与独立 [n]
    text = re.sub(r"[\[（(]\d+[\s.,，、-]*(?:\d+[,，]?)*[）\])]", " ", text)
    text = re.sub(r"[\[\]0-9]+", " ", text)  # 残余孤立编号
    text = re.sub(r"[（(][^（）()]{0,30}?原.*?[）)]", " ", text)
    text = re.sub(r"[（(][^（）()]{0,40}?存于互联网档案馆[）)]", " ", text)
    return text


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[。！？!?；;])\s*", text)
    return [s.strip() for s in sents if len(s.strip()) >= 20]


def chunk_sentences(sents: list[str]) -> list[str]:
    chunks, cur = [], ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= MAX_LEN and (len(cur) >= MIN_LEN or len(cur) + 1 + len(s) <= TARGET + 60):
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur.strip():
        chunks.append(cur)
    return chunks


def load_seeds():
    rows = [json.loads(l) for l in open(SEED_FILE, encoding="utf-8") if l.strip()]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--priority", default="A")
    parser.add_argument("--dry", action="store_true", help="只验证清洗，不写文件")
    parser.add_argument("--seed", help="仅处理指定 title_seed（dry-run 常用）")
    args = parser.parse_args()

    seeds = [r for r in load_seeds() if r["priority"] == args.priority]
    if args.seed:
        seeds = [r for r in seeds if r["title_seed"] == args.seed]

    stats = {"ok": 0, "missing": 0, "empty": 0, "total_chars": 0}
    skipped = []
    all_chunks = []  # (seed, chunk, resolved_title, domain)

    for i, seed in enumerate(seeds, 1):
        tid = seed["id"]
        title_seed = seed["title_seed"]
        try:
            resolved = resolve_title(title_seed)
            if not resolved:
                stats["missing"] += 1
                skipped.append((tid, title_seed, "title not found"))
                print(f"[{i}/{len(seeds)}] {tid} {title_seed!r} MISSING")
                continue
            extract = fetch_extract(resolved)
        except Exception as exc:  # noqa: BLE001
            stats["missing"] += 1
            skipped.append((tid, title_seed, f"fetch err: {exc!r}"))
            print(f"[{i}/{len(seeds)}] {tid} {title_seed!r} ERR: {exc!r}")
            time.sleep(2)
            continue

        if not extract.strip():
            stats["empty"] += 1
            skipped.append((tid, title_seed, "empty extract"))
            print(f"[{i}/{len(seeds)}] {tid} {resolved!r} EMPTY")
            continue

        cleaned = clean_text(extract)
        if len(cleaned) < TARGET:
            stats["empty"] += 1
            skipped.append((tid, title_seed, "too short after clean"))
            continue

        chunks = chunk_sentences(split_sentences(cleaned))
        for c in chunks:
            all_chunks.append((seed, c, resolved, seed["domain"]))
        stats["ok"] += 1
        stats["total_chars"] += len(cleaned)
        print(f"[{i}/{len(seeds)}] {tid} {resolved!r} ok, cleaned={len(cleaned)} chars, {len(chunks)} chunks")
        time.sleep(0.2)

    print("=" * 60)
    print(f"成功 {stats['ok']}  缺失 {stats['missing']}  空/过短 {stats['empty']}")
    print(f"清洗后总字符 {stats['total_chars']}  分块总数 {len(all_chunks)}")

    if args.dry:
        print("\n--- 清洗效果抽样（前3篇首块） ---")
        shown = set()
        for seed, c, resolved, _ in all_chunks:
            if seed["title_seed"] not in shown:
                shown.add(seed["title_seed"])
                print(f"\n>>> {resolved} (原文种子: {seed['title_seed']}, domain={seed['domain']})")
                print(c[:200])
                if len(shown) >= 3:
                    break
        if skipped:
            print("\nSKIPPED:")
            for tid, title, why in skipped:
                print(f"  {tid} {title}: {why}")
        return

    if not all_chunks:
        print("无可用数据，不写入。")
        sys.exit(0)

    # 从文件尾部续接 id
    next_num = 1
    from pathlib import Path  # noqa: PLC0415
    out_path = Path(OUT_FILE)
    if out_path.exists():
        last = 0
        for line in out_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            last = max(last, int(rec["id"].rsplit("-", 1)[1]))
        next_num = last + 1

    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for seed, c, resolved, domain in all_chunks:
            record = {
                "id": f"article-zh-{next_num:06d}",
                "split": "train",
                "language": "zh",
                "type": "article",
                "domain": domain,
                "topic": seed["title_seed"],
                "text": c,
                "source": "zh.wikipedia.org",
                "source_title": resolved,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            next_num += 1
            written += 1
    print(f"共追加 {written} 条 -> {OUT_FILE}")


if __name__ == "__main__":
    main()