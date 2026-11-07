#!/usr/bin/env python3
"""Inventory source pages and rights signals without assigning legal conclusions."""

from __future__ import annotations

import collections
import concurrent.futures
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "staging/round11/train"
DEST = ROOT / "reports/source_rights_inventory_round13.jsonl"
HEADERS = {"User-Agent": "littleGPT-source-audit/1.0 (local provenance audit)"}
POLICIES = {
    "www.energy.gov": "https://www.energy.gov/cmei/communicationstandards/copyright-laws-web",
    "www.nist.gov": "https://www.nist.gov/copyrights-disclaimers",
    "www.usgs.gov": "https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted",
    "science.nasa.gov": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
    "oceanservice.noaa.gov": "https://oceanservice.noaa.gov/about/faq.html",
    "zh.wikipedia.org": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en",
}
SIGNAL = re.compile(r"\b(?:copyright|license|credit|courtesy|permission)\b|版权|许可|授權|授权|署名", re.I)


def get_json(url: str) -> dict:
    error = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.load(response)
        except Exception as exc:
            error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"MediaWiki API failed after retries: {error}")


def wiki_batch(titles: list[str]) -> dict[str, dict]:
    params = {"action": "query", "titles": "|".join(titles), "redirects": "1", "converttitles": "1", "prop": "revisions", "rvprop": "ids|timestamp", "format": "json", "formatversion": "2"}
    result = get_json("https://zh.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params))["query"]
    converted = {item["from"]: item["to"] for item in result.get("converted", [])}
    redirects = {item["from"]: item["to"] for item in result.get("redirects", [])}
    normalized = {item["from"]: item["to"] for item in result.get("normalized", [])}
    pages = {page["title"]: page for page in result["pages"]}
    output = {}
    for title in titles:
        resolved = normalized.get(title, title)
        resolved = converted.get(resolved, resolved)
        resolved = redirects.get(resolved, resolved)
        page = pages.get(resolved, {})
        output[title] = {"current_page_title": page.get("title"), "current_page_id": page.get("pageid"), "current_revision_id": (page.get("revisions") or [{}])[0].get("revid"), "current_revision_timestamp": (page.get("revisions") or [{}])[0].get("timestamp"), "title_converted_or_redirected": title != resolved, "page_found": "missing" not in page and bool(page.get("pageid"))}
    return output


def check_page(url: str) -> dict:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as response:
            response.read(1024)  # Confirm that a page, not only a redirect, can be read.
            return {"page_http_status": response.status, "resolved_url": response.url}
    except Exception as exc:
        return {"page_http_status": None, "page_check_error": f"{type(exc).__name__}: {exc}"[:160]}


def main() -> None:
    english = collections.defaultdict(list)
    chinese = collections.defaultdict(list)
    generated = []
    for line in (SOURCE / "article_en.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        (english[row["source_url"]] if row.get("source_url") else generated).append(row)
    for line in (SOURCE / "article_zh.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        chinese[row["source_title"]].append(row)

    wiki = {}
    titles = sorted(chinese)
    for i in range(0, len(titles), 20):
        wiki.update(wiki_batch(titles[i : i + 20]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        page_checks = dict(zip(sorted(english), pool.map(check_page, sorted(english))))

    inventory = []
    for url, rows in sorted(english.items()):
        host = urllib.parse.urlsplit(url).netloc
        inventory.append({"source_kind": "external_english_page", "source_url": url, "source_title": rows[0].get("source_title"), "record_count": len(rows), "record_ids": [r["id"] for r in rows], "host": host, "site_policy_url": POLICIES[host], "rights_status": "site_policy_found_page_specific_terms_unconfirmed", "text_rights_signal_ids": [r["id"] for r in rows if SIGNAL.search(r["text"])], **page_checks[url]})
    for title, rows in sorted(chinese.items()):
        page = wiki[title]
        candidate = "https://zh.wikipedia.org/wiki/" + urllib.parse.quote(page["current_page_title"] or title)
        inventory.append({"source_kind": "external_chinese_wikipedia_page", "source_title": title, "candidate_current_url": candidate, "record_count": len(rows), "record_ids": [r["id"] for r in rows], "host": "zh.wikipedia.org", "site_policy_url": POLICIES["zh.wikipedia.org"], "rights_status": "current_page_found_original_revision_and_attribution_unconfirmed" if page["page_found"] else "title_unresolved_requires_source_match", "text_rights_signal_ids": [r["id"] for r in rows if SIGNAL.search(r["text"])], **page})
    DEST.parent.mkdir(exist_ok=True)
    with DEST.open("w", encoding="utf-8") as stream:
        for item in inventory:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"english_pages": len(english), "english_http_200": sum(p.get("page_http_status") == 200 for p in page_checks.values()), "chinese_titles": len(chinese), "chinese_pages_found": sum(p["page_found"] for p in wiki.values()), "chinese_unresolved_titles": [t for t, p in wiki.items() if not p["page_found"]], "ai_generated_english_rows": len(generated), "english_signal_rows": sum(len(x["text_rights_signal_ids"]) for x in inventory if x["source_kind"] == "external_english_page"), "chinese_signal_rows": sum(len(x["text_rights_signal_ids"]) for x in inventory if x["source_kind"] == "external_chinese_wikipedia_page")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
