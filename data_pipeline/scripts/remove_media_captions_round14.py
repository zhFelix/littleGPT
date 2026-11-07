#!/usr/bin/env python3
"""Remove five source-verified media captions from provisional split text."""

import hashlib
import json
import re
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "staging/round12"
DEST = ROOT / "staging/round14"
DECISIONS = ROOT / "reports/cleaning_decisions_round14.jsonl"
MANIFEST = ROOT / "reports/staging_round14_manifest.json"
TOKENIZER = Path("/Users/zhangyunjian/felixspace/rat/gpt2/tokenizer/tokenizer.json")
REMOVALS = {
    "article-en-000215": "Researchers observed the new phases in superconductor interactions, which could help build more robust superconductors. Credit: JILA/Steven Burrows ",
    "article-en-000277": "Download the illustration. Credit: NASA, ESA, CSA, Joseph Olmsted (STScI). ",
    "article-en-000321": "The LZ (LUX-ZEPLIN) liquid xenon-based dark matter experiment operating at the Sanford Underground Research Facility (SURF), nearly 1 mile underground in Lead, South Dakota. Photo courtesy of Matthew Kapust, Sanford Underground Research Facility ",
    "article-en-000413": "Credit: Blakestad/NIST ",
    "article-en-000636": "Video Url Scientists at Lawrence Berkeley National Laboratory wrote on this penny with an ion beam to show off how cool nanoscience research can be. Video courtesy of Lawrence Berkeley National Laboratory ",
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER))
    tok.no_truncation()
    decisions = []
    manifest = {"source": str(SOURCE), "status": "provisional_pending_public_release_rights_review", "rules": ["R14-remove-page-media-caption", "R14-annotate-ai-origin"], "files": {}, "changed_ids": [], "ai_origin_annotated": 0}
    ids = set()
    normalized_texts = set()
    group_splits = {}
    for split in ("train", "valid", "test"):
        (DEST / split).mkdir(parents=True, exist_ok=True)
        for name in ("article_en.jsonl", "article_zh.jsonl", "natural_en.jsonl", "natural_zh.jsonl"):
            src, dst = SOURCE / split / name, DEST / split / name
            count = tokens = 0
            with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
                for line_no, line in enumerate(inp, 1):
                    row = json.loads(line)
                    assert row["split"] == split and row["id"] not in ids
                    ids.add(row["id"])
                    assert group_splits.setdefault(row["group_id"], split) == split
                    if name in ("natural_en.jsonl", "natural_zh.jsonl"):
                        assert not row.get("origin_class")
                        before = json.dumps(row, ensure_ascii=False, sort_keys=True)
                        row["origin_class"] = "AI-generated"
                        row["origin_evidence"] = "user confirmation in current chat, 2026-11-07"
                        after = json.dumps(row, ensure_ascii=False, sort_keys=True)
                        decisions.append({"id": row["id"], "source_file": str(src), "source_line": line_no, "rule": "R14-annotate-ai-origin", "decision": "add origin_class and origin_evidence from user confirmation; text unchanged", "before_record_sha256": sha(before), "after_record_sha256": sha(after)})
                        manifest["ai_origin_annotated"] += 1
                    if row["id"] in REMOVALS:
                        old = row["text"]
                        fragment = REMOVALS[row["id"]]
                        assert old.count(fragment) == 1, row["id"]
                        row["text"] = old.replace(fragment, "", 1)
                        assert row["text"].strip()
                        decisions.append({"id": row["id"], "source_file": str(src), "source_line": line_no, "rule": "R14-remove-page-media-caption", "decision": "remove media-caption text and credit from training body; retain original in round12", "source_url": row.get("source_url"), "before_sha256": sha(old), "after_sha256": sha(row["text"]), "removed_text": fragment})
                        manifest["changed_ids"].append(row["id"])
                    key = re.sub(r"\s+", "", row["text"].casefold())
                    assert key not in normalized_texts, f"duplicate after caption removal: {row['id']}"
                    normalized_texts.add(key)
                    tokens += len(tok.encode(row["text"], add_special_tokens=False).ids)
                    count += 1
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["files"][f"{split}/{name}"] = {"records": count, "tokens": tokens, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    assert set(manifest["changed_ids"]) == set(REMOVALS)
    assert manifest["ai_origin_annotated"] == 8002
    with DECISIONS.open("w", encoding="utf-8") as out:
        for item in decisions:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(ids), "groups": len(group_splits), "caption_removed": manifest["changed_ids"], "ai_origin_annotated": manifest["ai_origin_annotated"], "tokens": sum(v["tokens"] for v in manifest["files"].values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
