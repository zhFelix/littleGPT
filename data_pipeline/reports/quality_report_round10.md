# 第 10 轮：原文核对与剩余疑点清洗

日期：2026-11-07

## 结果

- 对第 9 轮隔离的 51 条逐条核对原文，修复并恢复 51 条：中文百科 47 条、美国能源部英文 4 条。
- 本轮待核对队列剩余 0 条。第 10 轮暂存集共 10,862 条、910,385 个 tokenizer token。
- 第 9 轮已有的 10,811 条记录按 JSON 对象比较全部保持不变。四个 JSONL 文件均可解析；记录 ID 唯一、正文非空、规范化正文无重复。
- 这是清洗暂存版；外部文章的使用授权仍需按整体数据策略处理。

## 修复依据

各记录的原文 URL、页面行号、修复前后正文及 SHA-256 见 `source_repair_decisions_round10.jsonl`。按来源分组如下：

| 来源 | 条数 | 主要修复 |
| --- | ---: | --- |
| [电场](https://zh.wikipedia.org/wiki/电场) | 10 | 库仑力、电场积分、能量密度、延迟时间等公式 |
| [向量](https://zh.wikipedia.org/wiki/向量) | 6 | 范数、加法、内积、外积、分点坐标 |
| [积分](https://zh.wikipedia.org/wiki/积分) | 5 | 黎曼积分、积分性质、赫尔德与闵可夫斯基不等式 |
| [函数](https://zh.wikipedia.org/wiki/函数) | 4 | 定义域、竖直线判别、限制和函数运算 |
| [特征值和特征向量](https://zh.wikipedia.org/wiki/特征值和特征向量) | 4 | 幂法、谱、薛定谔方程、惯性张量 |
| [概率论](https://zh.wikipedia.org/wiki/概率论) | 4 | 全概率、贝叶斯公式及原文例题 |
| [势能](https://zh.wikipedia.org/wiki/势能) | 3 | 引力、弹性势能、静电力 |
| [导数](https://zh.wikipedia.org/wiki/导数) | 2 | 导数极限与高阶导数记法 |
| 其他中文条目 | 9 | 角动量、数列、矩阵、秩、空间复杂度、图、递归、反向传播、正则化的公式和标记 |
| [美国能源部科学说明](https://www.energy.gov/science) | 4 | 删除资源导航串入的正文，恢复其中一个上标 |

其中 `article-zh-000359` 的原页面把 `D_y` 集合的元素写成 `x`，上下文表明应为 `y`；修复日志已明确标为对原文疑似笔误的更正。`article-zh-000182` 的弹簧力公式按原页面保留 `F(x)=kx`，日志注明原文未区分回复力的负号约定。`article-zh-001040` 将原页面较长的嵌套链式推导压缩为等价的逐层梯度表达式。

## 交付文件

- `staging/round10/train/`：四类训练 JSONL。
- `review/quarantined_records_round10.jsonl`：空文件，表示本轮内容疑点清零。
- `reports/source_repair_decisions_round10.jsonl`：51 条逐条决策与前后正文。
- `reports/staging_round10_manifest.json`：计数、token 与文件 SHA-256。
- `scripts/reconcile_original_round10.py`：可复现脚本。
