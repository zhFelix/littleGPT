# littleGPT

一个使用中英文小语料训练轻量级 GPT-2 模型的实验项目。

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

除了 `text` 外，其他字段当前不会参与训练，但适合后续做数据分析或筛选。

### 2. TXT

纯文本格式，每行一条样本。

## 目录约定

- `train/`：训练集
- `valid/`：验证集
- `test/`：测试集

训练脚本会优先把 `valid/` 作为验证集；如果 `valid/` 不存在，则回退使用 `test/`。

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
python train_GPT2.py
```

常用参数示例：

```bash
python train_GPT2.py --epochs 30 --batch-size 8 --block-size 96 --learning-rate 5e-5
```

训练过程中会生成：

- `checkpoints/`
- `checkpoints/best/`
- `trained_model/`

### 4. 文本生成

```bash
python generate.py
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
python test.py
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
- `checkpoints/`：周期性保存的训练 checkpoint
- `checkpoints/best/`：验证集上当前最优模型
- `trained_model/`：训练结束后导出的最终模型

这些目录已经被 `.gitignore` 忽略，属于训练产物，不属于源码。

## 当前已知注意事项

1. 脚本当前只会自动读取 `chinese` 和 `english` 这两个数据集名。
2. `generate.py` 和 `test.py` 默认读取 `checkpoints/best`，所以第一次生成前需要先完成训练。
3. 这是一个实验型仓库，适合快速迭代，但还没有做成完整的配置化训练框架。
