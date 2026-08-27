#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the USGS "Science for Everyone" corpus from markdown dumps fetched by WebFetch.

Why: the live USGS site is behind AWS WAF (returns 405 + captcha page for plain
HTTP clients), so build_usgs_sfe_5_corpus.py cannot fetch pages directly. This
script instead converts saved markdown dumps into the same JSONL outputs, using
the exact same paragraph-cleaning + chunking logic (make_chunks from
build_usgs_sfe_5_corpus.py).

Usage:
    python build_usgs_corpus_from_dump.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201c\u2018(])")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def load_tokenizer(path: str | None):
    if not path:
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    tok.model_max_length = 10_000_000
    return tok


def token_count(text: str, tokenizer) -> int:
    if tokenizer is None:
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

        now = "\n\n".join(current).strip()
        if token_count(now, tokenizer) >= target_tokens:
            chunks.append(now)
            current = []

    if current:
        tail = "\n\n".join(current).strip()
        if chunks and token_count(tail, tokenizer) < min_tokens:
            merged = chunks[-1] + "\n\n" + tail
            if token_count(merged, tokenizer) <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(tail)
        else:
            chunks.append(tail)

    return [c for c in chunks if c.strip()]

# (source_id, title, domain, topic, url, dump_path)
ARTICLES = [
    (
        "usgs-sfe-001",
        "Using Distant Seismometers to Monitor and Analyze Volcanic Eruptions",
        "earth_science",
        "remote_volcano_seismic_monitoring",
        "https://www.usgs.gov/programs/earthquake-hazards/science/using-distant-seismometers-monitor-and-analyze-volcanic",
        "/var/folders/s0/5gcz208s4bs77n68gtjq4nlh0000gn/T/trae/toolcall-output/8e4cda61-52c2-45ad-966f-0fc1ac5fa9fd.txt",
    ),
    (
        "usgs-sfe-002",
        "The 2023 National Seismic Hazard Model \u2013 What's Shaking?",
        "earth_science",
        "national_seismic_hazard_model",
        "https://www.usgs.gov/programs/earthquake-hazards/science/2023-national-seismic-hazard-model-whats-shaking",
        "/var/folders/s0/5gcz208s4bs77n68gtjq4nlh0000gn/T/trae/toolcall-output/e2145341-60df-4582-9a99-6cb6dd65101d.txt",
    ),
    (
        "usgs-sfe-003",
        "Listening to the Earth at the South Pole",
        "earth_science",
        "south_pole_seismic_station",
        "https://www.usgs.gov/programs/earthquake-hazards/science/listening-earth-south-pole",
        "/var/folders/s0/5gcz208s4bs77n68gtjq4nlh0000gn/T/trae/toolcall-output/cdb633fd-1474-4a16-bd70-791165a34526.txt",
    ),
    (
        "usgs-sfe-004",
        "Improving Earthquake Monitoring with Deep Learning",
        "computer_science",
        "deep_learning_earthquake_monitoring",
        "https://www.usgs.gov/programs/earthquake-hazards/science/improving-earthquake-monitoring-deep-learning",
        "/var/folders/s0/5gcz208s4bs77n68gtjq4nlh0000gn/T/trae/toolcall-output/91d38e42-2145-480b-8de4-b740c3a67148.txt",
    ),
    (
        "usgs-sfe-005",
        "The Blind Zone of Earthquake Early Warning",
        "earth_science",
        "earthquake_early_warning_blind_zone",
        "https://www.usgs.gov/programs/earthquake-hazards/science/blind-zone-earthquake-early-warning",
        "/var/folders/s0/5gcz208s4bs77n68gtjq4nlh0000gn/T/trae/toolcall-output/66627ca0-c99e-4737-b289-3ea8e8413693.txt",
    ),
]

MIN_TOKENS = 60
TARGET_TOKENS = 78
MAX_TOKENS = 90

STOP_HEADINGS = {
    "for more information",
    "the scientist behind the science",
    "the scientists behind the science",
    "contacts",
    "related content",
    "related science",
    "publications",
    "data",
    "news",
    "explore search",
}
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
    "written by",
    "by ",
    "label",
)
CAPTION_SUFFIX_RE = re.compile(r"\(public domain\.\)\s*$", re.I)
CAPTION_FROM_RE = re.compile(r"\(from .+, \d{4}\)\s*$")
STRAY_FILE_REF_RE = re.compile(r"\[[^\]]*\.(?:pdf|jpg|jpeg|png)\b[^\]]*\]")
# Curly quotes and the okina -> plain ASCII (keeps the small vocab tokenizer clean).
NORMALIZE_MAP = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u02bb": "'",  # okina
}


def to_plain_text(line: str) -> str:
    """Strip WebFetch markdown/HTML wrappers down to plain text."""
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # [text](url) -> text
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)  # images
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)  # bold
    line = re.sub(r"\*([^*]+?)\*", r"\1", line)  # italic
    line = re.sub(r"\s+,", ",", line)  # "zone , an area" -> "zone, an area"
    line = re.sub(r"<sup>([^<]*)</sup>", r"\1", line)
    line = re.sub(r"<sub>([^<]*)</sub>", r"\1", line)
    line = re.sub(r"<u>([^<]*)</u>", r"\1", line)
    line = re.sub(r"<[^>]+>", "", line)  # any leftover tags
    line = re.sub(r"\\([~\-])", r"\1", line)  # escaped ~ / - from WebFetch
    line = STRAY_FILE_REF_RE.sub("", line)  # [F5a.pdf]
    line = line.translate(str.maketrans(NORMALIZE_MAP))
    # Remaining non-ASCII typography -> plain ASCII.
    line = line.replace("km\u00b2", "square kilometers").replace("mi\u00b2", "square miles")
    line = line.replace("\u2013", "-").replace("\u00b0", " degrees")
    line = line.replace("\u2026", "...").replace("\u00ae", "")
    line = SPACE_RE.sub(" ", line)
    return line.strip()


def extract_paragraphs(dump_path: str) -> list[str]:
    text = Path(dump_path).read_text(encoding="utf-8")
    paragraphs: list[str] = []
    seen: set[str] = set()
    stop = False
    media_pending = False

    for raw in text.splitlines():
        line = to_plain_text(raw)
        if not line:
            continue

        low = line.lower().strip(" :")

        # Image / "Sources/Usage:" lines open a media block; the next text
        # line is its caption, which must not enter the body.
        if line.startswith("![") or low.startswith("sources/usage:"):
            media_pending = True
            continue

        # Markdown headings never enter the training text; some act as
        # section boundaries that stop body collection.
        if line.startswith("#"):
            heading_low = line.lstrip("#").strip().lower().strip(" :")
            if heading_low in STOP_HEADINGS:
                stop = True
            continue

        # Page chrome: numbered nav and TOC/list links.
        if re.match(r"^\d+\.", line) or line.startswith("- "):
            continue

        # Status lines, and the bare "Overview" marker that starts the
        # duplicated accordion body.
        if low in {"completed", "active", "science", "media", "overview"}:
            if low == "overview":
                stop = True
            continue

        # Non-heading section markers (e.g. bold "**For More Information**").
        if low in STOP_HEADINGS:
            stop = True
            continue

        if stop:
            continue

        # Media caption directly following an image / sources block.
        if media_pending:
            media_pending = False
            continue

        if any(low.lstrip("~ ").startswith(prefix) for prefix in DROP_PREFIXES):
            continue

        # Media captions that leaked out of <figure> into the markdown dump.
        if line.startswith(("\u2013 ", "\u2014 ")):  # – / — dash leads
            continue
        if CAPTION_SUFFIX_RE.search(line) or CAPTION_FROM_RE.search(line):
            continue

        if len(line) < 35:
            continue

        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        paragraphs.append(line)

    return paragraphs


def main() -> None:
    tokenizer = load_tokenizer(str(BASE_DIR / "tokenizer"))

    output_rows = []
    raw_rows = []

    for src_id, title, domain, topic, url, dump_path in ARTICLES:
        paragraphs = extract_paragraphs(dump_path)
        if not paragraphs:
            print(f"WARNING: no paragraphs extracted from {dump_path}")
            continue

        raw_text = "\n\n".join(paragraphs)
        raw_rows.append({
            "id": src_id,
            "split": "train",
            "language": "en",
            "domain": domain,
            "topic": topic,
            "title": title,
            "source": "U.S. Geological Survey",
            "source_url": url,
            "text": raw_text,
        })

        chunks = make_chunks(
            paragraphs,
            tokenizer=tokenizer,
            min_tokens=MIN_TOKENS,
            target_tokens=TARGET_TOKENS,
            max_tokens=MAX_TOKENS,
        )

        for chunk_index, text in enumerate(chunks, start=1):
            output_rows.append({
                "id": f"{src_id}-{chunk_index:02d}",
                "split": "train",
                "language": "en",
                "domain": domain,
                "topic": topic,
                "text": text,
                "source": "U.S. Geological Survey",
                "source_url": url,
                "source_title": title,
                "chunk_index": chunk_index,
            })

        print(f"{src_id} {title!r}: paragraphs={len(paragraphs)} chunks={len(chunks)}")

    out_path = BASE_DIR / "usgs_sfe_5_clean_original.jsonl"
    raw_path = BASE_DIR / "usgs_sfe_5_clean_original_fulltext.jsonl"
    train_path = BASE_DIR / "train" / "article.jsonl"

    for path, rows in ((out_path, output_rows), (raw_path, raw_rows), (train_path, output_rows)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote chunks: {out_path} ({len(output_rows)} rows)")
    print(f"Wrote cleaned full text: {raw_path} ({len(raw_rows)} rows)")
    print(f"Copied chunks to train dataset: {train_path} ({len(output_rows)} rows)")


if __name__ == "__main__":
    main()
