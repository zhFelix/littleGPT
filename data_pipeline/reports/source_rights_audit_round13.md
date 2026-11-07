# 第 13 轮：外部文章来源与使用条件核查

日期：2026-11-07。核查对象为第 11 轮保留的 2,660 条外部文章记录；另有 200 条 `article_en` 已由用户确认为 ChatGPT 生成，不归入外部文章来源。逐来源记录、记录 ID、现存页面版本及正文中的权利关键词见 `source_rights_inventory_round13.jsonl`。这份清单记录证据，不向训练 JSONL 推断填入 `license`。

| 来源 | 页面/标题组 | 记录数 | 核查结果 |
|---|---:|---:|---|
| 中文维基百科 | 126 | 2,018 | MediaWiki API 均找到现存页面和当前修订号；16 个标题经简繁转换或重定向解析。原采集版本号未保存。 |
| 美国能源部 | 20 | 299 | 20 个页面可访问。 |
| 美国地质调查局 | 6 | 159 | 6 个页面经网页搜索或打开确认；自动 HTTP 请求返回 405，故清单保留该访问限制。 |
| 美国国家标准与技术研究院 | 3 | 124 | 3 个页面可访问。 |
| NASA Science | 2 | 49 | 2 个页面可访问。 |
| NOAA National Ocean Service | 2 | 11 | 2 个页面可访问。 |

## 站点规则与本批语料的关系

- [维基媒体使用条款](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en)及[内容再利用说明](https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse)说明维基百科文字通常按 CC BY-SA 4.0 共享，复用时需保留来源、作者归属、许可和修改说明；个别内容可能有例外。现有语料保存标题但未保存原修订号，公开发布时仍需准备可追溯的署名信息。
- [DOE](https://www.energy.gov/cmei/communicationstandards/copyright-laws-web)、[USGS](https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted)、[NIST](https://www.nist.gov/copyrights-disclaimers)、[NOAA](https://oceanservice.noaa.gov/about/faq.html)的官方说明均将机构自有内容与第三方受保护内容区分；来源域名本身不足以给每条记录认定为公有领域。各站点还提出来源标注要求或建议。
- [NASA 媒体使用指南](https://www.nasa.gov/nasa-brand-center/images-and-media/)提醒第三方素材即使显示在 NASA 网站上，也不随 NASA 的使用权一并授权。本批数据仅保存文字，但有一条正文保留了插图署名。
- [OpenAI 服务条款](https://openai.com/policies/row-terms-of-use/)规定用户与 OpenAI 之间的输出归属，同时指出输出可能不独特、第三方权利不在该归属范围内。200 条 ChatGPT 生成记录因此继续标明 AI 来源，不据此推断其事实或独创性。

## 需要单独处理的记录

英文正文中有 5 条保留了图片或视频的署名/来源说明：`article-en-000215`、`article-en-000277`、`article-en-000321`、`article-en-000413`、`article-en-000636`。这些文字说明不能当作整篇文章的许可声明；应核对它们是否属于待训练正文，若保留则维持原始署名。中文正文关键词扫描命中 8 条，但人工查看后均是正文谈论软件许可、访问授权等词义，未见页面复用许可声明。

中文维基百科有 6 个简体标题直接查询时显示为缺页；启用 MediaWiki 标题转换后均解析到繁体标题。将这 6 个现存页面导言转换为简体后，与对应语料开头逐字比对，相似度约为 0.87–1.00，支持标题对应关系；不代表已经找回采集时的精确版本。

## 尚需的使用场景信息

站点通用政策和现存页面已核对。最终发布方案仍取决于模型或语料是否公开、是否商业使用，以及是否有额外采集或授权记录。在明确这些条件前，第 12 轮三集合维持 `staging` 状态，不标作最终可训练版。
