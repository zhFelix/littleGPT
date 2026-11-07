# 公开发布准备状态（第 15 轮）

日期：2026-11-07。用户计划公开发布，但尚未决定发布模型、语料，还是两者。本文件同时准备两种路径；没有执行发布。

## 已完成

- 暂存语料为 10,861 条、904,265 tokenizer token；固定三集合和来源组，见 `quality_report_round15.md`。
- 2,659 条外部文章对应 159 个来源组；逐页清单见 `source_rights_inventory_round13.jsonl`。另 8,202 条记录标为 AI 生成。
- 已核对来源网站的一般政策，准备 `PUBLICATION_ATTRIBUTION_DRAFT_round15.md`，列出逐页链接、中文维基百科作者历史入口及清洗改动。
- 媒体署名和网页控件残留已从训练正文清理；原文与逐条日志保留在早期轮次及 `cleaning_decisions_round14/15.jsonl`。

## 如果只发布模型

发布材料应包含训练数据来源概况、AI 生成记录说明、模型与评测信息，以及数据处理和限制说明。需要先完成固定条件训练 A/B 和独立测试集评测。模型权重是否触发来源文字许可的义务，不能仅由站点通用政策推断；发布前应按实际发布方式单独审查。

## 如果发布语料

逐条文字会被再分发，需将来源与修改说明随语料提供。中文维基百科文字按其具体页面许可处理，通常涉及 [CC BY-SA 4.0](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en) 的署名、许可链接和修改说明；媒体与例外内容单独核查。美国政府网站的政策也明确区分机构自有内容与第三方内容，见 [DOE](https://www.energy.gov/cmei/communicationstandards/copyright-laws-web)、[USGS](https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted)、[NIST](https://www.nist.gov/copyrights-disclaimers)、[NOAA](https://oceanservice.noaa.gov/about/faq.html)和 [NASA](https://www.nasa.gov/nasa-brand-center/images-and-media/)。当前清单保存现存页面和修订信息，未保存采集时的精确版本，因此完整外部文章语料尚不能标为发布就绪。

## 当前关口

1. 保持 `staging/round15/` 为暂存版，不作为公开发布包。
2. 若最终发布语料，继续核对外部文章的采集版本、页面级第三方标记和署名要求；未核实的来源组可从发布包排除，但需记录决定。
3. 若最终发布模型，先完成训练 A/B 和测试集评测，并对拟发布模型做独立的来源权利审查。

公开范围尚未确定时，以上两个路径并行保留；不要求现在作选择。
