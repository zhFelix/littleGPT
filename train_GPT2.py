from pathlib import Path
import argparse
import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "train"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
OUTPUT_DIR = BASE_DIR / "trained_model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
BEST_CHECKPOINT_DIR = CHECKPOINT_DIR / "best"
DATASETS = ["chinese", "english"]

BLOCK_SIZE = 96
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-5
MODEL_EMBED_DIM = 384
MODEL_LAYER_COUNT = 6
MODEL_HEAD_COUNT = 6
SAVE_CHECKPOINT_EVERY = 20
EVAL_EVERY = 1
EARLY_STOPPING_PATIENCE = 0
MIN_IMPROVEMENT = 0.0


class TextBlockDataset(Dataset):
    def __init__(self, input_ids: list[int], block_size: int) -> None:
        self.samples = []
        if len(input_ids) < block_size + 1:
            raise ValueError("文本太短，无法切出训练样本。请增大语料或减小 BLOCK_SIZE。")

        for start in range(0, len(input_ids) - block_size, block_size):
            block = input_ids[start : start + block_size + 1]
            if len(block) == block_size + 1:
                self.samples.append(block)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        block = self.samples[index]
        input_ids = torch.tensor(block[:-1], dtype=torch.long)
        labels = torch.tensor(block[1:], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_model_config(tokenizer: AutoTokenizer, block_size: int) -> GPT2Config:
    return GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=block_size,
        n_embd=MODEL_EMBED_DIM,
        n_layer=MODEL_LAYER_COUNT,
        n_head=MODEL_HEAD_COUNT,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def build_model(tokenizer: AutoTokenizer, block_size: int) -> GPT2LMHeadModel:
    return GPT2LMHeadModel(build_model_config(tokenizer, block_size))


def get_model_architecture(config: GPT2Config) -> dict[str, int]:
    return {
        "n_embd": int(config.n_embd),
        "n_layer": int(config.n_layer),
        "n_head": int(config.n_head),
    }


def collect_dataset_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for dataset_name in DATASETS:
        jsonl_path = data_dir / f"{dataset_name}.jsonl"
        txt_path = data_dir / f"{dataset_name}.txt"
        if jsonl_path.exists():
            files.append(jsonl_path)
        elif txt_path.exists():
            files.append(txt_path)
    if not files:
        raise FileNotFoundError(f"No dataset files found in {data_dir}")
    return files


def collect_training_files() -> list[Path]:
    return collect_dataset_files(DATA_DIR)


def load_text_from_files(files: list[Path]) -> str:
    texts: list[str] = []
    for path in files:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if "text" not in record:
                    raise ValueError(f"{path} 中的记录缺少 text 字段。")
                texts.append(record["text"])
        else:
            texts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return "\n\n".join(texts)


def load_training_text(files: list[Path]) -> str:
    return load_text_from_files(files)


def choose_block_size(input_ids: list[int], preferred_block_size: int) -> int:
    max_allowed = len(input_ids) - 1
    if max_allowed < 4:
        raise ValueError("文本太短，至少需要更多 token 才能训练。")
    return min(preferred_block_size, max_allowed)


def get_checkpoint_path(epoch: int) -> Path:
    return CHECKPOINT_DIR / f"epoch_{epoch:03d}"


def find_latest_checkpoint() -> Path | None:
    if not CHECKPOINT_DIR.exists():
        return None

    checkpoints = sorted(
        (path for path in CHECKPOINT_DIR.glob("epoch_*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not checkpoints:
        return None
    return checkpoints[-1]


def save_checkpoint(
    checkpoint_path: Path,
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    block_size: int,
    extra_state: dict[str, object] | None = None,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    trainer_state: dict[str, object] = {
        "epoch": epoch,
        "block_size": block_size,
        "vocab_size": tokenizer.vocab_size,
        "model_n_embd": int(model.config.n_embd),
        "model_n_layer": int(model.config.n_layer),
        "model_n_head": int(model.config.n_head),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_state:
        trainer_state.update(extra_state)
    torch.save(trainer_state, checkpoint_path / "trainer_state.pt")
    print(f"Checkpoint saved to: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    expected_block_size: int,
    expected_vocab_size: int,
) -> tuple[GPT2LMHeadModel, dict[str, object], bool]:
    trainer_state = torch.load(checkpoint_path / "trainer_state.pt", map_location=device)
    checkpoint_block_size = int(trainer_state["block_size"])
    if checkpoint_block_size > expected_block_size:
        raise ValueError(
            f"Checkpoint block_size={checkpoint_block_size} 大于当前 block_size={expected_block_size}，"
            "暂不支持缩小位置嵌入，请删除旧 checkpoint 后重新训练。"
        )
    checkpoint_vocab_size = int(trainer_state.get("vocab_size", expected_vocab_size))
    if checkpoint_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size={checkpoint_vocab_size} 与当前 tokenizer vocab_size={expected_vocab_size} 不一致，请删除旧 checkpoint 后重新训练。"
        )

    checkpoint_config = GPT2Config.from_pretrained(checkpoint_path)
    checkpoint_architecture = {
        "n_embd": int(trainer_state.get("model_n_embd", checkpoint_config.n_embd)),
        "n_layer": int(trainer_state.get("model_n_layer", checkpoint_config.n_layer)),
        "n_head": int(trainer_state.get("model_n_head", checkpoint_config.n_head)),
    }
    expected_architecture = {
        "n_embd": MODEL_EMBED_DIM,
        "n_layer": MODEL_LAYER_COUNT,
        "n_head": MODEL_HEAD_COUNT,
    }
    if checkpoint_architecture != expected_architecture:
        raise ValueError(
            "Checkpoint model architecture is incompatible with the current training configuration: "
            f"checkpoint={checkpoint_architecture}, expected={expected_architecture}."
        )

    checkpoint_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    if checkpoint_block_size == expected_block_size:
        return checkpoint_model.to(device), trainer_state, False

    print(
        f"Expanding position embeddings from {checkpoint_block_size} to {expected_block_size} "
        f"using checkpoint: {checkpoint_path}"
    )
    model = expand_position_embeddings(checkpoint_model, expected_block_size)
    return model.to(device), trainer_state, True


def expand_position_embeddings(
    checkpoint_model: GPT2LMHeadModel,
    new_block_size: int,
) -> GPT2LMHeadModel:
    old_weight = checkpoint_model.transformer.wpe.weight.detach()
    old_block_size, hidden_size = old_weight.shape
    if new_block_size <= old_block_size:
        raise ValueError("new_block_size 必须大于旧的位置嵌入大小。")

    new_config = GPT2Config(**checkpoint_model.config.to_dict())
    new_config.n_positions = new_block_size
    new_config.n_ctx = new_block_size
    new_model = GPT2LMHeadModel(new_config)

    state_dict = checkpoint_model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key != "transformer.wpe.weight" and ".attn.bias" not in key and ".attn.masked_bias" not in key
    }
    incompatible_keys = new_model.load_state_dict(filtered_state_dict, strict=False)
    unexpected_keys = list(incompatible_keys.unexpected_keys)
    missing_keys = [
        key
        for key in incompatible_keys.missing_keys
        if key != "transformer.wpe.weight" and ".attn.bias" not in key and ".attn.masked_bias" not in key
    ]
    if unexpected_keys or missing_keys:
        raise ValueError(
            "Expanded checkpoint load found unexpected parameter mismatches: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )

    expanded_weight = old_weight.new_empty(new_block_size, hidden_size)
    expanded_weight[:old_block_size] = old_weight

    # Preserve the original learned positions exactly, then interpolate only the extra rows.
    target_positions = torch.linspace(0, old_block_size - 1, steps=new_block_size, device=old_weight.device)[old_block_size:]
    left_indices = torch.floor(target_positions).long()
    right_indices = torch.clamp(left_indices + 1, max=old_block_size - 1)
    interpolation_ratio = (target_positions - left_indices.to(target_positions.dtype)).unsqueeze(1)
    expanded_weight[old_block_size:] = (
        old_weight[left_indices] * (1.0 - interpolation_ratio) + old_weight[right_indices] * interpolation_ratio
    )

    with torch.no_grad():
        new_model.transformer.wpe.weight.copy_(expanded_weight)

    return new_model


def resolve_eval_dir(eval_dir_arg: str | None) -> Path | None:
    if eval_dir_arg:
        return Path(eval_dir_arg).expanduser().resolve()

    valid_dir = BASE_DIR / "valid"
    if valid_dir.exists():
        return valid_dir

    test_dir = BASE_DIR / "test"
    if test_dir.exists():
        return test_dir

    return None


def build_dataset_from_text(
    tokenizer: AutoTokenizer,
    text: str,
    block_size: int,
) -> TextBlockDataset | None:
    encoded = tokenizer(text, add_special_tokens=True, return_attention_mask=False)
    input_ids = encoded["input_ids"]
    if len(input_ids) < block_size + 1:
        return None
    return TextBlockDataset(input_ids, block_size)


def evaluate_model(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
    model.train()
    return total_loss / len(dataloader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or resume a small GPT-2 model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Target total epochs after resuming.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE, help="Preferred token block size.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="AdamW learning rate.")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory used for validation. Defaults to ./valid, or falls back to ./test when ./valid is missing.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=EVAL_EVERY,
        help="Run evaluation every N epochs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop after N evaluation rounds without improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=MIN_IMPROVEMENT,
        help="Minimum eval loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--save-checkpoint-every",
        type=int,
        default=SAVE_CHECKPOINT_EVERY,
        help="Save a checkpoint every N epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0。")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0。")
    if args.block_size <= 0:
        raise ValueError("--block-size 必须大于 0。")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate 必须大于 0。")
    if args.eval_every <= 0:
        raise ValueError("--eval-every 必须大于 0。")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience 不能小于 0。")
    if args.min_improvement < 0:
        raise ValueError("--min-improvement 不能小于 0。")
    if args.save_checkpoint_every <= 0:
        raise ValueError("--save-checkpoint-every 必须大于 0。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    # We chunk the token ids ourselves below, so the full corpus can exceed model_max_length.
    tokenizer.model_max_length = 10_000_000
    training_files = collect_training_files()
    text = load_training_text(training_files)
    training_input_ids = tokenizer(text, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    effective_block_size = choose_block_size(training_input_ids, args.block_size)
    training_dataset = TextBlockDataset(training_input_ids, effective_block_size)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Training files: {len(training_files)}")
    for file_path in training_files:
        print(f" - {file_path}")
    print(f"Training samples: {len(training_dataset)}, block_size: {effective_block_size}")

    eval_dir = resolve_eval_dir(args.eval_dir)
    eval_dataloader: DataLoader | None = None
    if eval_dir is not None:
        try:
            eval_files = collect_dataset_files(eval_dir)
            eval_text = load_text_from_files(eval_files)
            eval_dataset = build_dataset_from_text(tokenizer, eval_text, effective_block_size)
            if eval_dataset is None:
                print(f"Evaluation skipped: {eval_dir} 中的文本长度不足以切出 block_size={effective_block_size} 的样本。")
            else:
                eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
                print(f"Evaluation files: {len(eval_files)}")
                for file_path in eval_files:
                    print(f" - {file_path}")
                print(f"Evaluation samples: {len(eval_dataset)}")
        except FileNotFoundError:
            print(f"Evaluation skipped: no dataset files found in {eval_dir}")
    else:
        print("Evaluation skipped: neither ./valid nor ./test exists.")

    latest_checkpoint = BASE_DIR / "checkpoints" / "best"
    if not latest_checkpoint.exists():
        latest_checkpoint = find_latest_checkpoint()
    start_epoch = 0
    best_eval_loss: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    position_embeddings_expanded = False

    if latest_checkpoint is None:
        model = build_model(tokenizer, effective_block_size).to(device)
        print("No checkpoint found, training from scratch.")
    else:
        try:
            model, trainer_state, position_embeddings_expanded = load_checkpoint(
                latest_checkpoint,
                device,
                effective_block_size,
                tokenizer.vocab_size,
            )
            start_epoch = int(trainer_state["epoch"])
            saved_best_eval_loss = trainer_state.get("best_eval_loss")
            if saved_best_eval_loss is not None:
                best_eval_loss = float(saved_best_eval_loss)
            best_epoch = int(trainer_state.get("best_epoch", start_epoch))
            epochs_without_improvement = int(trainer_state.get("epochs_without_improvement", 0))
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        except ValueError as exc:
            print(f"Checkpoint skipped: {exc}")
            latest_checkpoint = None
            model = build_model(tokenizer, effective_block_size).to(device)
            print("Training from scratch with the current smaller model configuration.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if latest_checkpoint is not None and not position_embeddings_expanded:
        optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
    elif position_embeddings_expanded:
        print("Skipping optimizer state restore because position embeddings were expanded for this resume.")

    for param_group in optimizer.param_groups:
        param_group["lr"] = args.learning_rate

    if start_epoch >= args.epochs:
        print(
            f"Checkpoint has already finished {start_epoch} epochs. "
            f"Increase --epochs above {start_epoch} to continue training."
        )

    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={avg_loss:.4f}")

        current_epoch = epoch + 1
        should_evaluate = eval_dataloader is not None and (
            current_epoch % args.eval_every == 0 or current_epoch == args.epochs
        )
        if should_evaluate:
            eval_loss = evaluate_model(model, eval_dataloader, device)
            print(f"Epoch {current_epoch}/{args.epochs}, eval_loss={eval_loss:.4f}")

            is_improved = best_eval_loss is None or (best_eval_loss - eval_loss) > args.min_improvement
            if is_improved:
                best_eval_loss = eval_loss
                best_epoch = current_epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    BEST_CHECKPOINT_DIR,
                    model,
                    tokenizer,
                    optimizer,
                    current_epoch,
                    effective_block_size,
                    {
                        "best_eval_loss": best_eval_loss,
                        "best_epoch": best_epoch,
                        "epochs_without_improvement": epochs_without_improvement,
                        "is_best_checkpoint": True,
                    },
                )
                print(f"New best checkpoint: {BEST_CHECKPOINT_DIR} (epoch={best_epoch}, eval_loss={best_eval_loss:.4f})")
            else:
                epochs_without_improvement += 1
                print(
                    f"No eval improvement for {epochs_without_improvement} evaluation round(s). "
                    f"Current best: epoch={best_epoch}, eval_loss={best_eval_loss:.4f}"
                )

        trainer_state = {
            "best_eval_loss": best_eval_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        if current_epoch % args.save_checkpoint_every == 0 or current_epoch == args.epochs:
            save_checkpoint(
                get_checkpoint_path(current_epoch),
                model,
                tokenizer,
                optimizer,
                current_epoch,
                effective_block_size,
                trainer_state,
            )

        if (
            args.early_stopping_patience > 0
            and eval_dataloader is not None
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping triggered at epoch {current_epoch}. "
                f"Best checkpoint remains epoch={best_epoch}, eval_loss={best_eval_loss:.4f}"
            )
            break

    OUTPUT_DIR.mkdir(exist_ok=True)
    if eval_dataloader is not None and best_eval_loss is not None and BEST_CHECKPOINT_DIR.exists():
        best_model = AutoModelForCausalLM.from_pretrained(BEST_CHECKPOINT_DIR)
        best_tokenizer = AutoTokenizer.from_pretrained(BEST_CHECKPOINT_DIR)
        best_model.save_pretrained(OUTPUT_DIR)
        best_tokenizer.save_pretrained(OUTPUT_DIR)
        print(
            f"Best model saved to: {OUTPUT_DIR} "
            f"(source={BEST_CHECKPOINT_DIR}, epoch={best_epoch}, eval_loss={best_eval_loss:.4f})"
        )
    else:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Latest model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
