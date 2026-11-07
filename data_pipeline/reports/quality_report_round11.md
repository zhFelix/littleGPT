# 第 11 轮清洗报告：审核结论、标点空格与 topic 英译

日期：2026-11-07

## 结果

- 已读取 seed `20261107` 的 200 条人工审核结论：188 条通过、12 条需修改、0 条删除。12 条均要求 `natural_zh` 的 `topic` 用英文。
- `natural_zh` 全部 934 条中文 `topic` 已逐条译成英文，共 932 个不同标签；原值保存在新增字段 `topic_zh`。英文译名与中文原值一一对应，见 `topic_translation_map_round11.jsonl`。
- 中文正文中，删除中文标点后水平空格：`natural_zh` 501 条记录、1,324 处；`article_zh` 1,855 条记录、10,794 处。
- 英文正文没有发现需要按本规则补空格的普通英文标点用例，因此保持不变。缩写（如 `U.S.`、`e.g.`）、小数、公式和闭合引号均未机械改写。
- 第 11 轮暂存集共 10,862 条、905,055 个 tokenizer token。记录数与第 10 轮相同；ID 唯一、正文非空，规范化正文无重复。

## 验证

- 四类 JSONL 全部可解析；第 10 轮与第 11 轮逐条 ID 对齐。
- 除 `natural_zh` 的 `topic` 与新增 `topic_zh` 外，元数据保持不变。
- 中文正文的改动仅为移除指定中文标点后的水平空格；英文正文逐条保持不变。
- 新暂存版没有残留“中文标点后紧跟水平空格”的目标模式。
- 原始及第 10 轮暂存数据均未覆盖。

## 文件

- `staging/round11/train/`：四类第 11 轮暂存 JSONL。
- `reports/topic_translation_map_round11.jsonl`：932 个中文 `topic` 与英文译名、记录数。
- `reports/punctuation_decisions_round11.jsonl`：逐条标点改动、前后正文和哈希。
- `reports/staging_round11_manifest.json`：记录数、token 数、文件哈希。
- `review/random_sample_seed20261107_final_decisions.jsonl`：200 条审核结论。

本轮仍为暂存版；后续还需按既定策略完成外部文章许可审查、同源分组划分与跨集合泄漏检查。
