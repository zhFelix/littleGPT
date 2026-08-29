from pathlib import Path
import json

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "train"
OUTPUT_DIR = BASE_DIR / "tokenizer"
DATASETS = ["chinese", "english", "article"]
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
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
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
