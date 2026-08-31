#!/usr/bin/env python3
"""手动抓取 USGS science-en-006（requests 被判 405、浏览器先被验证码墙拦截，
由用户手动过验证码后在浏览器取得正文），按同款 split/chunk 追加到 train/article.jsonl。"""
import json
import re

OUT_FILE = "train/article.jsonl"

TARGET = 230
MIN_LEN = 140
MAX_LEN = 380
MIN_SENT = 25

SRC = {
    "id": "science-en-006",
    "title": "Groundwater Flow and the Water Cycle",
    "url": "https://www.usgs.gov/water-science-school/science/groundwater-flow-and-water-cycle",
    "source": "U.S. Geological Survey",
    "domain": "earth_science",
    "topic": "groundwater_flow",
    "text": """Yes, water below your feet is moving all the time, but not like rivers flowing below ground. It's more like water in a sponge. Gravity and pressure move water downward and sideways underground through spaces between rocks. Eventually it emerges back to the land surface, into rivers, and into the oceans to keep the water cycle going.

You see water all around you every day as lakes, rivers, ice, snow and rain. There are also vast amounts of water that are unseen, water existing in the ground. And even though groundwater is unseen, it is moving below your feet right now. As part of the water cycle, groundwater is a major contributor to flow in many streams and rivers and has a strong influence on river and wetland habitats for plants and animals. People have been using groundwater for thousands of years and continue to use it today, largely for drinking water and irrigation. Life on Earth depends on groundwater just as it does on surface water.

Have you ever heard that there are rivers of water flowing underground? Let's debunk this, it is pretty much a myth. Even though there are some caverns, lava and ice tubes, and horizontal springs that can carry water, the vast majority of underground water occupies the spaces between rocks and subsurface material. Generally, water underground is more like water in a sponge. It occupies the spaces between soil and rock particles. At a certain depth below the land surface, the spaces between the soil and rock particles can be totally filled with water, resulting in an aquifer from which groundwater can be pumped and used by people.

Groundwater flows underground at different rates. Some of the precipitation that falls onto the land infiltrates into the ground to become groundwater. If the water meets the water table, below which the soil is saturated, it can move both vertically and horizontally. Water moving downward can also meet more dense and water-resistant non-porous rock and soil, which causes it to flow in a more horizontal fashion, generally towards streams, the ocean, or deeper into the ground.

Groundwater, like any other part of the water cycle, is never totally static. The direction and speed of groundwater movement is determined by the various characteristics of aquifers and confining layers of subsurface rocks, which water has a difficult time penetrating, in the ground. Water moving below ground depends on the permeability, how easy or difficult it is for water to move, and on the porosity, the amount of open space in the material, of the subsurface rock. If the rock has characteristics that allow water to move relatively freely through it, then groundwater can move significant distances in a number of days. But groundwater can also sink into deep aquifers where it takes thousands of years to move back into the environment, or even go into deep groundwater storage, where it might stay for much longer periods.

If an aquifer is under enough pressure, an artesian well tapping the aquifer can result in pressurized water shooting above the land surface. Artesian well water is not really different from non-artesian well water, but it comes to the surface in a different manner. In unconfined aquifers, water has simply infiltrated from the surface and saturated the subsurface material, and if people drill a well into an unconfined aquifer, they have to install a pump to push water to the surface. Confined aquifers have layers of rock above and below them that are not very permeable to water, and natural pressure in the aquifer can sometimes be enough to push water in a well above the land surface. Not all confined aquifers produce artesian water, but artesian pressure can force water to the surface with great pressure. Mainly, the company that bottles artesian well water doesn't have to go to the expense of installing a pump in its well.

Even though the amount of water locked up in groundwater is a small percentage of all of Earth's water, it represents a large percentage of total freshwater on Earth. About 1.7 percent of all of Earth's water is groundwater, and about 30.1 percent of freshwater on Earth occurs as groundwater. About 5,614,000 cubic miles, or 23,400,000 cubic kilometers, of groundwater exist on Earth, of which about 54 percent is saline, with the remaining about 46 percent being freshwater. Only 0.8 percent of all the water on Earth is fresh groundwater.""",
}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n\n", " ").replace("\n", " "))
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < MIN_SENT:
            continue
        if re.search(r"(^related links|^tags:|^subscribe|cookie|all rights reserved|share this|quiz|media details|sources/usage)", p, re.I):
            continue
        out.append(p)
    return out


def chunk_sentences(sents: list[str]) -> list[str]:
    chunks = []
    cur = ""
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


def main() -> None:
    chunks = chunk_sentences(split_sentences(SRC["text"]))
    records = [
        {
            "id": f"{SRC['id']}-{idx:02d}",
            "split": "train",
            "language": "en",
            "domain": SRC["domain"],
            "topic": SRC["topic"],
            "text": chunk,
            "source": SRC["source"],
            "source_url": SRC["url"],
            "source_title": SRC["title"],
            "chunk_index": idx,
        }
        for idx, chunk in enumerate(chunks, start=1)
    ]
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lens = [len(r["text"]) for r in records]
    print(f"science-en-006: {len(records)} 块, 追加到 {OUT_FILE}")
    print(f"  长度 min/中位/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")


if __name__ == "__main__":
    main()