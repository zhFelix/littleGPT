#!/usr/bin/env python3
"""从 superconductivity_sources_5.json 中读取 URL，抓取正文、切块并追加到 train/article.jsonl。

切块风格参照现有 article.jsonl（中位数约 238 字符，多为 100-300）：
按句切分后贪心合并到目标长度，保留自然句子边界。
"""
import json
import re
import sys

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "Chrome/120.0 Safari/537.36"
}
SOURCES_FILE = "article/article.json"
OUT_FILE = "train/article.jsonl"

TARGET = 230     # 目标块长度
MIN_LEN = 140    # 至少多长才闭合一个块
MAX_LEN = 380    # 一块最多多长
MIN_SENT = 25    # 丢弃过短的句子片段


def fetch_article(url: str) -> tuple[str, str]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    for node in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "form"]):
        node.decompose()
    art = soup.find("article") or soup.find("main") or soup.body
    text = " ".join(art.get_text(" ", strip=True).split()) if art else ""
    return title, text


def split_sentences(text: str) -> list[str]:
    # 按句号/问号/感叹号后随空白或行分隔拆句
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < MIN_SENT:
            continue
        # 过滤导航/版权/分享类噪音（含常见无正文结尾词）
        if re.search(r"(^related links|^tags:|^subscribe|cookie|all rights reserved|share this)", p, re.I):
            continue
        # 过滤政府网站 boilerplate：页脚说明、跳转提示、通讯作者简介/联系方式、引用URL
        if re.search(
            r"(^DOE Explains offers straightforward|"
            r"^It also describes how these concepts apply|"
            r"Skip to main content|"
            r"official websites use \.gov|"
            r"writes and curates content|"
            r"communications specialist in the Vehicle|"
            r"For more information please visit|"
            r"Shannon Brescher Shea|"
            r"^https?://)",
            p,
            re.I,
        ):
            continue
        out.append(p)
    return out


def chunk_sentences(sents: list[str]) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= MAX_LEN and (len(cur) >= MIN_LEN or len(cur) + 1 + len(s) <= TARGET + 60):
            # 还没到目标时继续积累；到目标附近后只有在不超长时才并入
            cur = cur + " " + s
        else:
            chunks.append(cur)
            cur = s
    if cur.strip():
        chunks.append(cur)
    return chunks


def main() -> None:
    sources = json.load(open(SOURCES_FILE, encoding="utf-8"))
    records = []
    skipped = []

    for src in sources:
        sid = src["id"]
        url = src["url"]
        try:
            title, text = fetch_article(url)
        except Exception as exc:  # noqa: BLE001
            skipped.append((sid, f"fetch failed: {exc!r}"))
            continue
        if not text:
            skipped.append((sid, "empty text"))
            continue

        chunks = chunk_sentences(split_sentences(text))
        for idx, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "id": f"{sid}-{idx:02d}",
                    "split": "train",
                    "language": "en",
                    "domain": "physics",
                    "topic": src["topic"],
                    "text": chunk,
                    "source": src["source"],
                    "source_url": url,
                    "source_title": src["title"],
                    "chunk_index": idx,
                }
            )
        print(f"{sid}: {len(chunks)} chunks, text_len={len(text)}, title={title[:40]!r}")

    if skipped:
        print("SKIPPED:")
        for sid, why in skipped:
            print(f"  {sid}: {why}")

    if not records:
        print("没有任何可追加的块。")
        sys.exit(1)

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"共追加 {len(records)} 条 -> {OUT_FILE}")


if __name__ == "__main__":
    main()