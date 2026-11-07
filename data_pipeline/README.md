# littleGPT 数据清洗暂存版

`staging/round15/{train,valid,test}/` 是截至 2026-11-07 的最新清洗快照，共 10,861 条记录。`reports/staging_round15_manifest.json` 保存每个文件的记录数、token 数和 SHA-256；`reports/quality_report_round15.md` 记录本轮检查结果。清洗脚本、逐条决定与人工审核记录分别保存在 `scripts/`、`reports/` 和 `review/`。

这份语料仍处于暂存状态：外部文章的公开发布许可尚未逐页确认。来源清单和署名草稿见 `reports/source_rights_inventory_round13.jsonl` 与 `reports/PUBLICATION_ATTRIBUTION_DRAFT_round15.md`。

`train_GPT2.py` 当前默认读取仓库的 `data/train/`，不会自动读取本目录。恢复训练前应为原始版和清洗版分别建立独立的数据、检查点与模型输出目录，并使用相同的随机种子、初始化、训练步数和独立测试集；不要覆盖原始 `data/train/`。
