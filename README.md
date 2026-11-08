# littleGPT

一个使用中英文小语料训练轻量级 GPT-2 模型的简单项目。

这个仓库包含一个完整但简洁的流程：

1. 使用训练语料训练自定义 BPE 分词器
2. 基于该分词器从零训练一个小型 GPT-2
3. 从最佳 checkpoint 加载模型进行文本生成
4. 在测试集上做基础评测和 smoke test

## 项目目的

这个项目适合用来：

- 学习 GPT-2 的最小训练闭环
- 在小语料上快速验证文本生成实验
- 练习 tokenizer、训练、验证、测试的完整流程
- 作为后续优化训练效果或工程结构的基础仓库

## 项目结构

```text
littleGPT/
├─ train/                  # 训练数据
├─ valid/                  # 验证数据
├─ test/                   # 测试数据
├─ training_tokenizer.py   # 训练 tokenizer
├─ train_GPT2.py           # 训练或续训模型
├─ generate.py             # 使用最佳 checkpoint 做文本生成
├─ test.py                 # 测试集评测与生成测试
├─ load_GPT2.py            # 打印当前模型参数规模
├─ requirements.txt        # Python 依赖
├─ .gitignore
└─ README.md
```

## 核心脚本说明

### `training_tokenizer.py`

- 从 `train/` 目录读取训练语料
- 训练一个 BPE tokenizer
- 输出到 `tokenizer/`

默认使用的数据集名是：

- `chinese`
- `english`

也就是说，脚本会优先读取：

- `train/chinese.jsonl` 或 `train/chinese.txt`
- `train/english.jsonl` 或 `train/english.txt`

### `train_GPT2.py`

- 加载 `tokenizer/`
- 读取 `train/` 数据
- 构建一个轻量 GPT-2 模型
- 支持保存 checkpoint、从 checkpoint 续训、验证集评估、early stopping
- 最终模型输出到 `trained_model/`

当前默认模型配置较小，适合实验：

- `block_size = 96`
- `batch_size = 4`
- `epochs = 20`
- `n_embd = 384`
- `n_layer = 6`
- `n_head = 6`

### `generate.py`

- 默认从 `checkpoints/best` 读取模型
- 输入一个 prompt，输出生成结果
- 支持采样生成和贪心生成

### `test.py`

- 从 `checkpoints/best` 读取模型
- 在 `test/` 数据集上计算平均 loss 和 perplexity
- 对若干 prompt 做文本生成测试

### `load_GPT2.py`

- 单独构建当前配置下的 GPT-2
- 打印模型参数量
- 适合快速了解模型规模

## 数据格式

项目支持两种语料格式：

### 1. JSONL

每行一个 JSON 对象，至少需要包含 `text` 字段。例如：

```json
{"id": "train-zh-0001", "split": "train", "language": "zh", "domain": "geography", "topic": "china_geography", "text": "中国的首都是北京。"}
```

除了 `text` 外，其他字段当前不会参与训练，但可以后续做数据分析或筛选。

### 2. TXT

纯文本格式，每行一条样本。

## 目录约定

- `train/`：训练集
- `valid/`：验证集
- `test/`：测试集

训练脚本只把 `valid/` 用作默认验证集；若它不存在，会在训练前报错。`test/` 不参与训练或 early stopping。

## 快速开始

### 1. 安装依赖

项目代码依赖以下 Python 库：

- `torch`
- `transformers`
- `tokenizers`

推荐直接安装 `requirements.txt`：

```bash
pip install -r requirements.txt
```

macOS Apple Silicon 上若旧环境导入 SciPy 的 `_spropack` 失败，可建立独立的 Python 3.12 环境：

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-macos-mps.txt
.venv/bin/python train_GPT2.py --help
```

已在 Apple M1 / macOS 27.0.1 验证上述依赖能导入训练脚本、加载 tokenizer，且非沙箱环境下 MPS 可用。默认的 `data/train/` 已复制清洗版 round15 训练数据；`valid/` 和 `test/` 保留旧模型使用的原始验证与测试数据。原训练数据备份在本地 `backups/pre_round15_20261108/`；round15 三集合快照仍在 `data_pipeline/staging/round15/`。

如果你需要手动安装，也可以使用：

```bash
pip install torch transformers tokenizers
```

### 2. 训练 tokenizer

```bash
python training_tokenizer.py
```

运行后会生成：

- `tokenizer/`

### 3. 训练模型

```bash
python train_GPT2.py --epochs 40 --early-stopping-patience 3
```

此命令从 `data/train/` 读取 round15 训练集、用原始 `valid/` 验证，默认写入新的 `runs/round15_old_eval/checkpoints/` 和 `runs/round15_old_eval/model/`。旧模型仍在 `checkpoints/` 和 `littleGPT_model/`。先前按 round15 验证集训练的 4 轮产物留在 `runs/round15/`，不会被续接。再次使用同一 checkpoint 目录会自动续训；如需另一组从零训练，请指定新的空目录。

其他训练参数示例：

```bash
python train_GPT2.py --epochs 30 --batch-size 8 --block-size 96 --learning-rate 5e-5 --early-stopping-patience 3 --save-checkpoint-every 5
```

若要改用其他训练数据，须同时指定训练集、验证集及独立的产物目录：

```bash
python train_GPT2.py \
  --train-dir path/to/other/train \
  --eval-dir path/to/other/valid \
  --checkpoint-dir runs/other/checkpoints \
  --output-dir runs/other/model \
  --epochs 40 --early-stopping-patience 3
```

`--train-dir` 与其余三个目录参数须一起提供，避免不同数据共用 checkpoint。本次对照沿用原始 `test/` 做最终评估，不用于 early stopping。
- --epochs：表示这次训练中要训练多少轮
- --batch-size：表示模型在一次前向传播（推理）或一次梯度更新（训练）中，同时处理的“样本/序列”的数量。
- --block-size：表示这次训练的上下文长度
- --learning-rate：表示这次训练的学习率
- --early-stopping-patience 3：表示在评估级重复多少次没有改善就停止训练
 - --save-checkpoint-every：表示这次训练训练多少轮保存一次

默认训练过程中会生成

- `runs/round15_old_eval/checkpoints/`
- `runs/round15_old_eval/checkpoints/best/`
- `runs/round15_old_eval/model/`

`runs/` 下的产物不会改动旧模型。

### 4. 文本生成

```bash
python generate.py --model-dir runs/round15_old_eval/model
```

自定义 prompt：

```bash
python generate.py "中国的首都是"
python generate.py "Machine learning is"
```

使用贪心解码：

```bash
python generate.py "中国的首都是" --greedy
```

### 5. 运行测试

```bash
python test.py --model-dir runs/round15_old_eval/checkpoints/best
```

也可以自定义 prompt：

```bash
python test.py --prompt "中国的首都是" --prompt "A useful scientific theory"
```

## 默认工作流

推荐按下面顺序运行：

```text
1. 准备 train/valid/test 数据
2. 运行 training_tokenizer.py
3. 运行 train_GPT2.py
4. 运行 generate.py 检查生成效果
5. 运行 test.py 查看 loss / perplexity 和生成样例
```

## 输出目录说明

- `tokenizer/`：训练得到的分词器
- `checkpoints/` 和 `littleGPT_model/`：上一轮训练产物，保留作对照
- `runs/round15_old_eval/checkpoints/best/`：新训练在原始验证集上最优的 checkpoint
- `runs/round15_old_eval/model/`：新训练结束后导出的最终模型
- `runs/round15/`：曾使用 round15 验证集的 4 轮部分训练产物
- `backups/pre_round15_20261108/`：覆盖前的本地训练和验证数据备份

`runs/` 和 `backups/` 已被 `.gitignore` 忽略，属于本地训练产物和备份。

## 注意事项

1. 训练脚本优先读取 `natural_zh`、`natural_en`、`article_zh`、`article_en`，未找到时读取目录中的 JSONL/TXT 文件。
2. `generate.py` 和 `test.py` 仍默认读取旧模型目录；检查新模型时请显式传入 `--model-dir`。`test.py` 读取原始 `test/` 数据，供新旧模型在同一测试集上比较。
3. 这是一个实验型仓库，适合快速迭代，但还没有做成完整的配置化训练框架。
4. 训练脚本按 CUDA → MPS → CPU 的顺序选择设备；请安装与当前设备兼容的 PyTorch。
