#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a clean USGS Science for Everyone training corpus.

Principle:
- Preserve USGS article wording and sentence order.
- Do NOT summarize, paraphrase, or "AI rewrite" the body text.
- Remove page chrome, media/captions, contact/profile blocks, references,
  repeated page fragments, and obvious non-body text.
- Split only at natural paragraph/sentence boundaries.
- If a local tokenizer is supplied, chunks are targeted to ~60-90 tokens.

Install:
    pip install requests beautifulsoup4

Optional for exact token-based chunking:
    pip install transformers tokenizers

Example:
    python build_usgs_sfe_5_corpus.py \
        --output usgs_sfe_5_clean_original.jsonl

With your tokenizer:
    python build_usgs_sfe_5_corpus.py \
        --tokenizer ./tokenizer \
        --min-tokens 60 \
        --target-tokens 78 \
        --max-tokens 90 \
        --output usgs_sfe_5_clean_original.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup, Tag

SOURCES = [
    {
        "id": "usgs-sfe-001",
        "title": "Using Distant Seismometers to Monitor and Analyze Volcanic Eruptions",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/science/using-distant-seismometers-monitor-and-analyze-volcanic",
        "domain": "earth_science",
        "topic": "remote_volcano_seismic_monitoring",
    },
    {
        "id": "usgs-sfe-002",
        "title": "The 2023 National Seismic Hazard Model – What's Shaking?",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/science/2023-national-seismic-hazard-model-whats-shaking",
        "domain": "earth_science",
        "topic": "national_seismic_hazard_model",
    },
    {
        "id": "usgs-sfe-003",
        "title": "Listening to the Earth at the South Pole",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/science/listening-earth-south-pole",
        "domain": "earth_science",
        "topic": "south_pole_seismic_station",
    },
    {
        "id": "usgs-sfe-004",
        "title": "Improving Earthquake Monitoring with Deep Learning",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/science/improving-earthquake-monitoring-deep-learning",
        "domain": "computer_science",
        "topic": "deep_learning_earthquake_monitoring",
    },
    {
        "id": "usgs-sfe-005",
        "title": "The Blind Zone of Earthquake Early Warning",
        "url": "https://www.usgs.gov/programs/earthquake-hazards/science/blind-zone-earthquake-early-warning",
        "domain": "earth_science",
        "topic": "earthquake_early_warning_blind_zone",
    },
]

USER_AGENT = "Mozilla/5.0 (compatible; littleGPT-corpus-builder/1.0)"

# Blocks/sections that are page chrome rather than article prose.
DROP_TAGS = {
    "script", "style", "nav", "footer", "form", "button", "svg",
    "noscript", "aside", "figure", "picture", "video", "audio",
    "blockquote",
}

# Text markers that usually begin non-body sections on USGS pages.
STOP_HEADINGS = {
    "for more information",
    "the scientist behind the science",
    "the scientists behind the science",
    "contacts",
    "contact",
    "related content",
    "related science",
    "publications",
    "data",
    "news",
    "explore search",
}

# Exact/near-exact page fragments to remove.
DROP_PREFIXES = (
    "sources/usage:",
    "view media details",
    "release date:",
    "credit:",
    "photo credit:",
    "image credit:",
    "learn more",
    "view all",
    "items per page",
    "label",
)

DROP_EXACT = {
    "media",
    "active",
    "completed",
    "science",
    "news",
    "publications",
    "data",
}

# Class/id keywords associated with page UI, media, contacts, related cards, etc.
DROP_CLASS_RE = re.compile(
    r"(media|caption|credit|contact|profile|person|author|sidebar|footer|"
    r"breadcrumb|social|share|related|views-field|pager|navigation|menu|"
    r"hero|card|teaser|field--name-field-media|field--type-entity-reference)",
    re.I,
)

SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘(])")

def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text

def element_is_noise(el: Tag) -> bool:
    classes = " ".join(el.get("class", []))
    ident = str(el.get("id", ""))
    marker = f"{classes} {ident}"
    return bool(DROP_CLASS_RE.search(marker))

def fetch(url: str, timeout: int = 30) -> str:
    r = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    r.raise_for_status()
    return r.text

def extract_body_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    for name in DROP_TAGS:
        for tag in soup.find_all(name):
            tag.decompose()

    # Prefer the central page content.
    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup

    # Remove obvious UI/media/contact/profile containers before extracting text.
    for el in list(main.find_all(True)):
        if element_is_noise(el):
            el.decompose()

    paragraphs: list[str] = []
    seen: set[str] = set()
    stop = False

    # p is the main source. h2/h3 are inspected only as section boundaries;
    # headings themselves are not inserted into training text.
    for el in main.find_all(["h2", "h3", "p"]):
        text = normalize_text(el.get_text(" ", strip=True))
        if not text:
            continue

        low = text.lower().strip(" :")

        if el.name in {"h2", "h3"}:
            if low in STOP_HEADINGS:
                stop = True
            # Do not put headings into the model text.
            continue

        if stop:
            continue

        if low in DROP_EXACT:
            continue
        if any(low.startswith(prefix) for prefix in DROP_PREFIXES):
            continue

        # Remove tiny UI fragments / orphan labels.
        if len(text) < 35:
            continue

        # Avoid obvious copyright/media-credit paragraphs.
        if re.search(
            r"(©|copyright|all rights reserved|photo by|image courtesy of|"
            r"view media|sources/usage)",
            text,
            re.I,
        ):
            continue

        # Exact paragraph dedup, preserving first occurrence.
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        paragraphs.append(text)

    return paragraphs

def load_tokenizer(path: str | None):
    if not path:
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    # We need counts, never truncation.
    tok.model_max_length = 10_000_000
    return tok

def token_count(text: str, tokenizer) -> int:
    if tokenizer is None:
        # Fallback heuristic for English prose.
        # The actual model tokenizer is preferred.
        return max(1, round(len(text.split()) * 1.35))
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def split_long_paragraph(paragraph: str, tokenizer, max_tokens: int) -> list[str]:
    if token_count(paragraph, tokenizer) <= max_tokens:
        return [paragraph]

    sentences = [
        normalize_text(x)
        for x in SENTENCE_SPLIT_RE.split(paragraph)
        if normalize_text(x)
    ]
    if len(sentences) <= 1:
        # Last resort: split on semicolon boundaries, still preserving wording.
        sentences = [
            normalize_text(x)
            for x in re.split(r"(?<=;)\s+", paragraph)
            if normalize_text(x)
        ]

    out: list[str] = []
    current: list[str] = []

    for sent in sentences:
        candidate = " ".join(current + [sent]).strip()
        if current and token_count(candidate, tokenizer) > max_tokens:
            out.append(" ".join(current).strip())
            current = [sent]
        else:
            current.append(sent)

    if current:
        out.append(" ".join(current).strip())

    return out

def make_chunks(
    paragraphs: list[str],
    tokenizer,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
) -> list[str]:
    # First ensure no paragraph exceeds max_tokens.
    units: list[str] = []
    for p in paragraphs:
        units.extend(split_long_paragraph(p, tokenizer, max_tokens))

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = "\n\n".join(current + [unit]).strip()
        cand_tokens = token_count(candidate, tokenizer)

        if current and cand_tokens > max_tokens:
            chunks.append("\n\n".join(current).strip())
            current = [unit]
            continue

        current.append(unit)

        # End naturally once we are around target size.
        now = "\n\n".join(current).strip()
        if token_count(now, tokenizer) >= target_tokens:
            chunks.append(now)
            current = []

    if current:
        tail = "\n\n".join(current).strip()
        # If tail is too short, merge it into the previous chunk only if max allows.
        if chunks and token_count(tail, tokenizer) < min_tokens:
            merged = chunks[-1] + "\n\n" + tail
            if token_count(merged, tokenizer) <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(tail)
        else:
            chunks.append(tail)

    return [c for c in chunks if c.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="usgs_sfe_5_clean_original.jsonl")
    ap.add_argument("--raw-output", default="usgs_sfe_5_clean_original_fulltext.jsonl")
    ap.add_argument("--tokenizer", default=None,
                    help="Path/name for Hugging Face tokenizer. Recommended: ./tokenizer")
    ap.add_argument("--min-tokens", type=int, default=60)
    ap.add_argument("--target-tokens", type=int, default=78)
    ap.add_argument("--max-tokens", type=int, default=90)
    ap.add_argument("--sleep", type=float, default=0.8)
    args = ap.parse_args()

    if not (1 <= args.min_tokens <= args.target_tokens <= args.max_tokens):
        raise SystemExit("Require min_tokens <= target_tokens <= max_tokens")

    tokenizer = load_tokenizer(args.tokenizer)

    output_rows = []
    raw_rows = []

    for article_index, src in enumerate(SOURCES, start=1):
        print(f"[{article_index}/{len(SOURCES)}] Fetching: {src['title']}")
        html = fetch(src["url"])
        paragraphs = extract_body_paragraphs(html)

        if not paragraphs:
            print("  WARNING: no body paragraphs extracted; inspect page manually.")
            continue

        raw_text = "\n\n".join(paragraphs)
        raw_rows.append({
            "id": src["id"],
            "split": "train",
            "language": "en",
            "domain": src["domain"],
            "topic": src["topic"],
            "title": src["title"],
            "source": "U.S. Geological Survey",
            "source_url": src["url"],
            "text": raw_text,
        })

        chunks = make_chunks(
            paragraphs,
            tokenizer=tokenizer,
            min_tokens=args.min_tokens,
            target_tokens=args.target_tokens,
            max_tokens=args.max_tokens,
        )

        for chunk_index, text in enumerate(chunks, start=1):
            output_rows.append({
                "id": f"{src['id']}-{chunk_index:02d}",
                "split": "train",
                "language": "en",
                "domain": src["domain"],
                "topic": src["topic"],
                "text": text,
                "source": "U.S. Geological Survey",
                "source_url": src["url"],
                "source_title": src["title"],
                "chunk_index": chunk_index,
                "token_count": token_count(text, tokenizer),
            })

        print(f"  kept paragraphs: {len(paragraphs)}")
        print(f"  chunks: {len(chunks)}")
        time.sleep(args.sleep)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    raw_path = Path(args.raw_output)
    with raw_path.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote chunks: {out_path} ({len(output_rows)} rows)")
    print(f"Wrote cleaned full text: {raw_path} ({len(raw_rows)} rows)")
    if tokenizer is None:
        print("NOTE: token_count is an English-word heuristic. Re-run with --tokenizer ./tokenizer")
        print("      for counts matched to your littleGPT tokenizer.")

if __name__ == "__main__":
    main()
