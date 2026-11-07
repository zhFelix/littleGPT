# littleGPT 外部文章来源与许可核查

日期：2026-11-07。本文件记录早期来源恢复线索；第 11 轮语料的最新逐来源核查见 `source_rights_audit_round13.md` 和 `source_rights_inventory_round13.jsonl`。本文件不是逐篇许可结论。

## 来源恢复结果

- `article_zh`：当前原始记录包含 `source=zh.wikipedia.org` 和 `source_title`；第六批暂存版的 1,927 条记录被归入 126 个标题组。另有 91 条公式/标记疑点暂时隔离。
- `article_en`：`article/` 目录的三个 JSON 文件列出 30 个来源网址。仓库历史 `173e319^:train/article.jsonl` 另外保留了旧记录的 `source_url`、`source_title` 和分块序号。按原文唯一匹配，当前原始语料 854 条中有 654 条可追溯；第八批暂存版 838 条中有 638 条已恢复逐条外部来源，涉及 33 个网址；其余 200 条没有历史匹配，现已由用户确认为 ChatGPT 生成，故无外部来源 URL。
- `article/dynamtic earth.txt` 是独立文本文件；当前 `article_en` 的原始正文没有与该文件逐字相同的完整记录，因此未据此给任何记录赋来源。
- 第六批只补充可核实的元数据和文章组 ID；第八批恢复 200 条已确认的 AI 生成记录。正文与顺序均未改写。逐条证据见 `provenance_restoration_log.jsonl` 和 `cleaning_decisions_round8.jsonl`。

| 英文文章来源 | 第八批记录数 | 不同网址数 |
|---|---:|---:|
| U.S. Department of Energy | 295 | 20 |
| U.S. Geological Survey | 159 | 6 |
| National Institute of Standards and Technology | 124 | 3 |
| NASA Science | 49 | 2 |
| NOAA | 11 | 2 |
| ChatGPT 生成（用户确认） | 200 | — |

## 公开使用政策线索

- [Wikimedia 内容再利用说明](https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse)说明维基百科文本通常使用 CC BY-SA 4.0，需核对具体页面并保留许可与署名信息。训练及模型输出的合规安排仍需单独判断。
- [USGS 版权说明](https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted)、[NOAA National Ocean Service FAQ](https://oceanservice.noaa.gov/about/faq.html)、[NIST 版权说明](https://www.nist.gov/copyrights-disclaimers)均区分机构自有内容与第三方标注内容。
- [DOE 网站版权说明](https://www.energy.gov/cmei/communicationstandards/copyright-laws-web)指出政府机构制作的内容通常属公有领域，但第三方版权内容需取得权利人许可。
- [NASA 媒体使用指南](https://www.nasa.gov/nasa-brand-center/images-and-media/)提醒第三方版权材料不随 NASA 网页一并授权。

这些是来源网站的一般规则，不能替代逐页检查。尤其是第二批从正文移出的图片署名与许可证文字已经保存在决策日志，不应丢失。

## 尚需补证

1. 外部文章在采集时的页面版本、第三方内容标记和实际使用范围；目前不能给全部记录写入确定的 `license` 值。
2. 第 10 轮已参考原文处理并放回先前隔离的 95 条公式/标记疑点；第 11 轮已落实 seed `20261107` 的 200 条随机抽检结论。外部文章的逐页许可核查仍未完成。
