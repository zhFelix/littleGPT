# littleGPT 公式核对与暂存结果（第九批）

2026-11-07 对照维基百科条目中的公式与上下文，修复并恢复 44 条中文文章记录。逐条的来源 URL、网页行号、修复前后正文和 SHA-256 见 `formula_repair_decisions_round9.jsonl`。网页行号对应本次检索结果，页面更新后可能变化。

- 第九批暂存训练集：10,811 条，904,675 个正文 Token（不含 EOS）。
- 44 条公式提取疑点均已修复；其它 51 条风险记录继续隔离。其标记数为：表格或标记片段 42、括号不平衡 6、网页导航混入 4（标记可重叠）。
- 第八批原有 10,767 条记录保持不变；第九批新增 44 条修复记录。四份 JSONL 可解析、正文非空、无跨文件精确重复。
- 原始语料未改动；第九批仍是暂存版，外部文章许可审查与随机质量抽检仍未完成。

## 对照原文

- [二分搜寻](https://zh.wikipedia.org/wiki/二分搜尋)：4 条
- [光](https://zh.wikipedia.org/wiki/光)：1 条
- [函数](https://zh.wikipedia.org/wiki/函数)：3 条
- [力](https://zh.wikipedia.org/wiki/力)：1 条
- [势能](https://zh.wikipedia.org/wiki/势能)：1 条
- [导数](https://zh.wikipedia.org/wiki/导数)：4 条
- [微分方程](https://zh.wikipedia.org/wiki/微分方程)：1 条
- [微积分学](https://zh.wikipedia.org/wiki/微积分学)：1 条
- [感知器](https://zh.wikipedia.org/wiki/感知器)：1 条
- [排序算法](https://zh.wikipedia.org/wiki/排序算法)：5 条
- [时间复杂度](https://zh.wikipedia.org/wiki/时间复杂度)：2 条
- [梯度下降法](https://zh.wikipedia.org/wiki/梯度下降法)：2 条
- [正则化 (数学)](https://zh.wikipedia.org/wiki/正则化_(数学))：1 条
- [波](https://zh.wikipedia.org/wiki/波)：1 条
- [电场](https://zh.wikipedia.org/wiki/电场)：1 条
- [矩阵](https://zh.wikipedia.org/wiki/矩阵)：7 条
- [积分](https://zh.wikipedia.org/wiki/积分)：2 条
- [空间复杂度](https://zh.wikipedia.org/wiki/空间复杂度)：2 条
- [简谐运动](https://zh.wikipedia.org/wiki/簡諧運動)：1 条
- [算法](https://zh.wikipedia.org/wiki/算法)：1 条
- [递归](https://zh.wikipedia.org/wiki/递归)：1 条
- [链表](https://zh.wikipedia.org/wiki/链表)：1 条
