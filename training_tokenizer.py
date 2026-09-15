from pathlib import Path
import json

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# 需要被 NFKC 折叠的中文全角标点/符号 -> 映射到私用区占位符(占位符不会被 NFKC 改变)
# 这样 NFKC 只折叠拉丁/数字/其他符号，中文全角标点保持不变，作为独立 token
CJK_PUNCT_PLACEHOLDERS = {
    "，": "\uE000",
    "：": "\uE001",
    "；": "\uE002",
    "！": "\uE003",
    "？": "\uE004",
    "（": "\uE005",
    "）": "\uE006",
    "％": "\uE007",
    "＃": "\uE008",
    "＠": "\uE009",
    "｛": "\uE00A",
    "｝": "\uE00B",
    "［": "\uE00C",
    "］": "\uE00D",
    "…": "\uE00E",
    "　": "\uE00F",
}


def build_normalizer() -> normalizers.Normalizer:
    """构建保留中文全角标点的归一化器。

    NFKC 会把中文全角标点(，：；！？、括号等)折叠成半角，导致模型学不到
    地道中文标点。这里在 NFKC 前把中文标点替换成私用区占位符标记（私用区
    不会被 NFKC 改变），NFKC 后再还原为占位符，从而让中文标点作为独立 token
    保留下来。
    """
    # 保护(替换为占位) -> NFKC -> 还原(占位替换回中文标点)
    norm_steps = []
    for zh, ph in CJK_PUNCT_PLACEHOLDERS.items():
        if zh != ph:
            norm_steps.append(normalizers.Replace(zh, ph))
    norm_steps.append(normalizers.NFKC())
    for zh, ph in CJK_PUNCT_PLACEHOLDERS.items():
        norm_steps.append(normalizers.Replace(ph, zh))
    return normalizers.Sequence(norm_steps)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "train"
OUTPUT_DIR = BASE_DIR / "tokenizer"
DATASETS = ["natural_zh", "natural_en", "article_zh", "article_en"]
# DATASETS = ["overfit_chinese", "overfit_english"]
VOCAB_SIZE = 10000
MODEL_MAX_LENGTH = 256
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
SAMPLE_TEXT = "Artificial intelligence is changing the world. 人工智能正在改变世界。"


def collect_training_files() -> list[Path]:
    files: list[Path] = []
    for dataset_name in DATASETS:
        jsonl_path = DATA_DIR / f"{dataset_name}.jsonl"
        txt_path = DATA_DIR / f"{dataset_name}.txt"
        if jsonl_path.exists():
            files.append(jsonl_path)
        elif txt_path.exists():
            files.append(txt_path)
    if not files:
        raise FileNotFoundError(f"No training files found in {DATA_DIR}")
    return files


def load_training_texts(files: list[Path]) -> list[str]:
    texts: list[str] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if "text" not in record:
                    raise ValueError(f"{file_path} 中的记录缺少 text 字段。")
                texts.append(record["text"])
        else:
            texts.extend(line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return texts


def train_tokenizer() -> None:
    training_files = collect_training_files()
    training_texts = load_training_texts(training_files)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = build_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(training_texts, trainer=trainer)

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        special_tokens=[
            ("<|bos|>", bos_id),
            ("<|eos|>", eos_id),
        ],
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        model_max_length=MODEL_MAX_LENGTH,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    fast_tokenizer.save_pretrained(OUTPUT_DIR)

    encoded = fast_tokenizer(SAMPLE_TEXT, add_special_tokens=True)

    print(f"Training files: {len(training_files)}")
    for file_path in training_files:
        print(f" - {file_path}")
    print(f"Training samples: {len(training_texts)}")
    print(f"Tokenizer saved to: {OUTPUT_DIR}")
    print(f"Vocab size: {fast_tokenizer.vocab_size}")
    print(f"BOS token id: {fast_tokenizer.bos_token_id}")
    print(f"EOS token id: {fast_tokenizer.eos_token_id}")
    print(f"Sample text: {SAMPLE_TEXT}")
    print(f"Input ids: {encoded['input_ids']}")
    print(f"Tokens: {fast_tokenizer.convert_ids_to_tokens(encoded['input_ids'])}")


if __name__ == "__main__":
    train_tokenizer()
