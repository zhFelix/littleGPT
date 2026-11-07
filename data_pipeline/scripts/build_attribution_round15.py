#!/usr/bin/env python3
"""Build a public-release attribution draft from verified source inventory."""

import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "reports/source_rights_inventory_round13.jsonl"
OUTPUT = ROOT / "reports/PUBLICATION_ATTRIBUTION_DRAFT_round15.md"
ACTIVE = ROOT / "staging/round15"
AGENCIES = {
    "zh.wikipedia.org": "中文维基百科贡献者",
    "www.energy.gov": "U.S. Department of Energy",
    "www.usgs.gov": "U.S. Geological Survey",
    "www.nist.gov": "National Institute of Standards and Technology",
    "science.nasa.gov": "NASA Science",
    "oceanservice.noaa.gov": "NOAA National Ocean Service",
}


def main() -> None:
    active_ids = set()
    for split in ("train", "valid", "test"):
        for path in (ACTIVE / split).glob("*.jsonl"):
            active_ids.update(json.loads(line)["id"] for line in path.open(encoding="utf-8"))
    rows = []
    for line in INVENTORY.open(encoding="utf-8"):
        row = json.loads(line)
        row["record_ids"] = [record_id for record_id in row["record_ids"] if record_id in active_ids]
        row["record_count"] = len(row["record_ids"])
        if row["record_count"]:
            rows.append(row)
    by_host = collections.defaultdict(list)
    for row in rows:
        by_host[row["host"]].append(row)
    lines = [
        "# 公开发布署名草稿（第 15 轮）",
        "",
        "此文件用于准备公开发布模型和/或语料；尚未完成逐页许可结论，不应直接当作发布许可。来源清单和证据见 `source_rights_inventory_round13.jsonl`。",
        "",
        "- 清洗改动：格式整理、去重、公式修复、标点整理、元数据英译、5 段媒体说明移除及第 15 轮网页杂质清理。逐条决定见各轮 `cleaning_decisions*.jsonl`。",
        "- 语料中另有 8,002 条 `natural_zh/en` 和 200 条 `article_en` 被标为 AI 生成。公开提供这些记录时，应保留 `origin_class` 和来源说明，不把它们列为外部文章。",
        "- 中文维基百科文字通常按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 提供。以下链接指向现存页面及作者历史；采集时的精确修订号未知。若发布修改后的文字，需要据具体许可保留署名、许可链接和修改说明。",
        "- 美国政府网站的机构自有内容与第三方素材须分开判断。以下列出的站点政策是审核依据，不能替代逐页确认或公开发布决定。",
        "",
    ]
    for host in ("zh.wikipedia.org", "www.energy.gov", "www.usgs.gov", "www.nist.gov", "science.nasa.gov", "oceanservice.noaa.gov"):
        group = sorted(by_host[host], key=lambda row: row["source_title"])
        lines.extend([f"## {AGENCIES[host]}：{len(group)} 个来源，{sum(x['record_count'] for x in group)} 条记录", "", f"站点政策：[查看]({group[0]['site_policy_url']})。", ""])
        if host == "zh.wikipedia.org":
            lines.extend(["| 来源页面 | 作者历史 | 记录数 |", "|---|---|---:|"])
            for row in group:
                url = row["candidate_current_url"]
                history = url + "?action=history"
                lines.append(f"| [{row['source_title']}]({url}) | [历史]({history}) | {row['record_count']} |")
        else:
            lines.extend(["| 来源页面 | 记录数 |", "|---|---:|"])
            for row in group:
                lines.append(f"| [{row['source_title']}]({row['source_url']}) | {row['record_count']} |")
        lines.append("")
    lines.extend(["## 发布前核对", "", "1. 确定最终发布范围：仅模型、仅语料，或两者。", "2. 对所有外部文章核查采集版本、页面上的第三方标记以及拟发布材料是否包含这些标记对应的内容。", "3. 核对中文维基百科及其他需署名内容的许可、作者归属和修改说明，并将本草稿随发布物提供。", "4. 根据实际发布范围决定是否排除尚未完成来源核查的记录；任何排除都应重新检查训练/验证/测试划分与评测。", ""])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"pages": len(rows), "records": sum(x["record_count"] for x in rows), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
