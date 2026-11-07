# 第 15 轮：网页杂质清理与发布署名更新

日期：2026-11-07。来源为第 14 轮三集合；沿用固定 seed `20261107` 的分组和集合归属。

| 集合 | 记录数 | tokenizer token |
|---|---:|---:|
| train | 8,690 | 713,437 |
| valid | 1,085 | 95,412 |
| test | 1,086 | 95,416 |
| 合计 | 10,861 | 904,265 |

- 逐条清理 16 条英文文章记录中的图片/视频引用、页面日期与阅读时间、图库导航、播放器提示、无法渲染占位文字及表格装饰标题。具体移除文本、来源行、规则和前后哈希见 `cleaning_decisions_round15.jsonl`。
- `article-en-000287` 仅含视频播放器提示、动画/音乐署名与下载入口，已从暂存训练集移除。第 14 轮原文仍保留，日志记有原文哈希和删除理由。
- 检查输出：10,861 个 ID 唯一、正文非空；规范化正文无完全重复；所有保留记录的 `group_id` 只在一个集合。验证与测试集的记录及 token 数没有变化。
- 公开发布署名草稿已按保留记录重算：`PUBLICATION_ATTRIBUTION_DRAFT_round15.md` 涵盖 159 个外部来源组、2,659 条记录。它仍是草稿，逐页许可和发布范围结论尚未完成。

复现命令：`python3 outputs/data_pipeline/scripts/remove_web_furniture_round15.py` 和 `python3 outputs/data_pipeline/scripts/build_attribution_round15.py`。输出位于 `outputs/data_pipeline/staging/round15/`。本轮仍是暂存版。
