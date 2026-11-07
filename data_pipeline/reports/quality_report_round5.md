# littleGPT 暂存语料质量报告（第五批）

第五批从第四批暂存训练集隔离 95 条待复核记录，保留完整原文与可恢复清单。训练集为 10,767 条、897,954 个正文 Token；原始语料未修改。

训练文件 JSONL 有效、正文非空且无精确重复。与 valid/test 的精确及字符 5-gram Jaccard ≥0.85 跨集合检查未发现重复；来源、许可及人工抽检尚待补齐。详见 `staging_round5_manifest.json` 与 `cleaning_decisions_round5.jsonl`。
