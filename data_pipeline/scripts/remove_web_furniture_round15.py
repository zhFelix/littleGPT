#!/usr/bin/env python3
"""Remove source-verified page navigation, widgets, and media captions."""

import hashlib
import json
import re
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "staging/round14"
DEST = ROOT / "staging/round15"
DECISIONS = ROOT / "reports/cleaning_decisions_round15.jsonl"
MANIFEST = ROOT / "reports/staging_round15_manifest.json"
TOKENIZER = "/Users/zhangyunjian/felixspace/rat/gpt2/tokenizer/tokenizer.json"
DROP = {"article-en-000287": "video player fallback, animation credit, and download widget only"}
REPLACEMENTS = {
    "article-en-000061": [(" (see image)", "")],
    "article-en-000162": [(" Office of Science March 1, 2017 Estimated Read Time min This figure shows how electrons pair up to cause superconductivity. Instead of traveling independently, the electrons couple into pairs that flow through metal without resistance.", "")],
    "article-en-000163": [("Image is taken from the Report of the Basic Energy Sciences Workshop on Superconductivity, May 8-11, 2006 ", "")],
    "article-en-000184": [("Image is taken from the Report of the Basic Energy Sciences Workshop on Superconductivity, May 8-11, 2006 A Shared Characteristic ", "")],
    "article-en-000248": [(" We have already found lots of superconductors, but this whimsical illustration shows why one superconductor's newfound properties may make it especially useful.", "")],
    "article-en-000275": [(" (see video above)", "")],
    "article-en-000276": [(" This illustration shows a quasar, an active supermassive black hole at the center of a galaxy. The black hole, which is represented by the black dot, is surrounded by a hot disk of gas, known as an accretion disk. Quasars periodically emit bipolar jets.", "")],
    "article-en-000288": [("Explore This Section Exoplanets ", ""), (" Unable to render the provided source 6K, and counting...", "")],
    "article-en-000307": [("‘We Are Seekers’: Watch the Video Exoplanet Types ", "")],
    "article-en-000317": [("NASA Exoplanets 17 Images Go To Gallery Go To Gallery Unable to render the provided source ", "")],
    "article-en-000337": [("Home Ocean Facts What is Ocean Acidification? What is Ocean Acidification? ", ""), (" What is ocean acidification? Watch our video for a quick overview.", "")],
    "article-en-000420": [("Qubit Contenders Qubit type Image Description Trapped ion ", "")],
    "article-en-000488": [("Image of the inside of a magnetic confinement experiment during plasma discharge. ", "")],
    "article-en-000524": [("An image of the debris left over after the creation of a quark-gluon plasma in the collision of two nuclei at Brookhaven National Laboratory. ", "")],
    "article-en-000560": [(", such as Earth in this image", "")],
    "article-en-000572": [(" Resources DOE SC Nuclear Physics program DOE SC High Energy Physics program Fermilab video: What’s relativity all about?", "")],
}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    tok = Tokenizer.from_file(TOKENIZER)
    tok.no_truncation()
    decisions = []
    seen_ids = set()
    normalized = set()
    groups = {}
    manifest = {"source": str(SOURCE), "status": "provisional_pending_public_release_rights_review", "rule": "R15-remove-web-furniture", "files": {}, "edited_ids": [], "dropped_ids": []}
    for split in ("train", "valid", "test"):
        (DEST / split).mkdir(parents=True, exist_ok=True)
        for name in ("article_en.jsonl", "article_zh.jsonl", "natural_en.jsonl", "natural_zh.jsonl"):
            src, dst = SOURCE / split / name, DEST / split / name
            count = tokens = 0
            with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
                for line_no, line in enumerate(inp, 1):
                    row = json.loads(line)
                    record_id = row["id"]
                    assert record_id not in seen_ids and row["split"] == split
                    seen_ids.add(record_id)
                    assert groups.setdefault(row["group_id"], split) == split
                    before = row["text"]
                    if record_id in DROP:
                        decisions.append({"id": record_id, "source_file": str(src), "source_line": line_no, "rule": "R15-remove-web-furniture", "decision": "drop", "reason": DROP[record_id], "before_sha256": sha(before), "source_url": row.get("source_url")})
                        manifest["dropped_ids"].append(record_id)
                        continue
                    if record_id in REPLACEMENTS:
                        removed = []
                        for old, new in REPLACEMENTS[record_id]:
                            assert row["text"].count(old) == 1, record_id
                            row["text"] = row["text"].replace(old, new, 1)
                            removed.append(old)
                        assert row["text"].strip()
                        decisions.append({"id": record_id, "source_file": str(src), "source_line": line_no, "rule": "R15-remove-web-furniture", "decision": "edit", "reason": "remove page navigation, media caption or widget text", "before_sha256": sha(before), "after_sha256": sha(row["text"]), "removed_fragments": removed, "source_url": row.get("source_url")})
                        manifest["edited_ids"].append(record_id)
                    normalized_text = re.sub(r"\s+", "", row["text"].casefold())
                    assert normalized_text not in normalized, record_id
                    normalized.add(normalized_text)
                    tokens += len(tok.encode(row["text"], add_special_tokens=False).ids)
                    count += 1
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["files"][f"{split}/{name}"] = {"records": count, "tokens": tokens, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    assert set(manifest["edited_ids"]) == set(REPLACEMENTS)
    assert set(manifest["dropped_ids"]) == set(DROP)
    with DECISIONS.open("w", encoding="utf-8") as out:
        for decision in decisions:
            out.write(json.dumps(decision, ensure_ascii=False) + "\n")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": sum(item["records"] for item in manifest["files"].values()), "tokens": sum(item["tokens"] for item in manifest["files"].values()), "edited": len(manifest["edited_ids"]), "dropped": manifest["dropped_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
