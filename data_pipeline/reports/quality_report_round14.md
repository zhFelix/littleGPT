# 第 14 轮：公开发布准备与媒体说明清理

日期：2026-11-07。依据用户计划公开发布模型或语料，本轮保留第 12 轮的固定三集合和来源组；没有移动、删除记录，也未填入未经确认的 `license`。

| 集合 | 记录数 | 第 14 轮 tokenizer token |
|---|---:|---:|
| train | 8,691 | 714,009 |
| valid | 1,085 | 95,412 |
| test | 1,086 | 95,416 |
| 合计 | 10,862 | 904,837 |

- 对照来源网页，移除 5 条英文训练记录中混入的图片/视频说明及媒体署名：`article-en-000215`、`article-en-000277`、`article-en-000321`、`article-en-000413`、`article-en-000636`。所有 5 条都在 train；正文减少 218 token。原文仍保存在第 12 轮，逐条移除片段和前后哈希见 `cleaning_decisions_round14.jsonl`。
- 按用户确认的来源，为全部 8,002 条 `natural_zh/en` 添加 `origin_class=AI-generated` 与 `origin_evidence`。正文未改。另 200 条 ChatGPT 生成的 `article_en` 在之前轮次已标记。
- 核对输出：ID 唯一、正文非空、无规范化完全重复、8,361 个 `group_id` 均只在一个集合；总记录数和各集合记录数均不变。
- 已生成 159 个外部文章来源组的 `PUBLICATION_ATTRIBUTION_DRAFT_round14.md`，包含中文维基百科的页面与历史链接，以及英文站点的逐页链接。它是署名草稿，不是发布许可结论。

复现命令：`python3 outputs/data_pipeline/scripts/remove_media_captions_round14.py` 和 `python3 outputs/data_pipeline/scripts/build_attribution_round14.py`。输出位于 `outputs/data_pipeline/staging/round14/`。

公开发布前仍需核对外部文章在采集时的版本与第三方内容，并明确最终发布范围。第 14 轮继续标为暂存版。
