# littleGPT 数据清洗暂存版

`staging/round15/{train,valid,test}/` 是截至 2026-11-07 的最新清洗快照，共 10,861 条记录。`reports/staging_round15_manifest.json` 保存每个文件的记录数、token 数和 SHA-256；`reports/quality_report_round15.md` 记录本轮检查结果。清洗脚本、逐条决定与人工审核记录分别保存在 `scripts/`、`reports/` 和 `review/`。

这份语料仍处于暂存状态：外部文章的公开发布许可尚未逐页确认。来源清单和署名草稿见 `reports/source_rights_inventory_round13.jsonl` 与 `reports/PUBLICATION_ATTRIBUTION_DRAFT_round15.md`。

第 12 轮将清洗后的四类语料按来源组及近重复候选划分为 train / valid / test，固定种子为 `20261107`，目标比例为 80% / 10% / 10%；第 15 轮沿用集合归属。原仓库的 `valid/chinese.jsonl` 和 `valid/english.jsonl` 是更早建立的 200 条验证集，与本目录的 round15 `valid/` 不是同一批记录。

仓库默认的 `data/train/` 已复制 round15 训练集合；`valid/` 和 `test/` 沿用旧模型的原始集合。覆盖前的训练文件备份在本地 `backups/pre_round15_20261108/`。`train_GPT2.py` 默认把新训练产物写入 `runs/round15_old_eval/`，保留旧 `checkpoints/` 和 `littleGPT_model/`。本目录的 round15 `valid/` 和 `test/` 仍保存为清洗快照，不用于当前新旧模型的对照评估。
