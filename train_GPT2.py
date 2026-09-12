from pathlib import Path
import argparse
import json
import math
import random
import shutil

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "train"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
OUTPUT_DIR = BASE_DIR / "littleGPT_model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
BEST_CHECKPOINT_DIR = CHECKPOINT_DIR / "best"
DATASETS = ["chinese", "english", "article", "qa_zh", "qa_en"]
# DATASETS = ["overfit_chinese", "overfit_english"]

BLOCK_SIZE = 96
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 2e-4
# WEIGHT_DECAY = 0.1
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
WARMUP_RATIO = 0.025
MODEL_EMBED_DIM = 384
MODEL_LAYER_COUNT = 6
MODEL_HEAD_COUNT = 6
MODEL_DROPOUT = 0.15  # 数据量小的场景下比默认 0.1 更抗过拟合；数据量大幅增加后仍可保留
SAVE_CHECKPOINT_EVERY = 20
EVAL_EVERY = 1
EARLY_STOPPING_PATIENCE = 0
MIN_IMPROVEMENT = 0.005
TRAINING_OBJECTIVE_VERSION = 2  # v2: labels=input_ids，由 GPT2LMHeadModel 内部完成单次 shift


class TextBlockDataset(Dataset):
    """固定窗口数据集，用于验证。

    labels 与 input_ids 完全相同。GPT2LMHeadModel 会在内部完成 causal LM 的
    一位 shift，因此这里不能提前把 labels 右移，否则会形成 double shift。
    """

    def __init__(self, input_ids: list[int], block_size: int) -> None:
        self.samples = []
        if len(input_ids) < block_size:
            raise ValueError("文本太短，无法切出训练样本。请增大语料或减小 BLOCK_SIZE。")

        for start in range(0, len(input_ids) - block_size + 1, block_size):
            block = input_ids[start : start + block_size]
            if len(block) == block_size:
                self.samples.append(block)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.samples[index], dtype=torch.long)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class RandomWindowDataset(Dataset):
    """对整条连续 token 流做随机窗口采样（token 级随机起点）。

    - 每条 token 在采样窗口中的可达概率一致 -> 天然按 token 占比均衡语料，
      所有文档（含短中文）都保留，不因 block_size 过滤而丢数据；
    - 同一数据流每轮以不同起点出现 -> 天然实现随机窗口数据增强，
      缓解"固定块每轮重复"导致的快速过拟合。
    """

    def __init__(self, input_ids: list[int], block_size: int, seed: int = 42) -> None:
        if len(input_ids) < block_size:
            raise ValueError("文本太短，无法切出训练窗口。请增大语料或减小 BLOCK_SIZE。")
        self.ids = torch.tensor(input_ids, dtype=torch.long)
        self.block_size = block_size
        self.num_samples = max(1, len(input_ids) // block_size)
        self.max_start = len(input_ids) - block_size
        self.epoch = 0
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # 3.11+ 的 random.Random.seed 不再接受 tuple，需组合成确定性 int 种子
        seed_int = (
            (self.seed * 73856093) ^ (self.epoch * 19349663) ^ (index * 83492791)
        )
        rng = random.Random(seed_int)
        start = rng.randint(0, self.max_start)
        input_ids = self.ids[start : start + self.block_size].clone()
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_model_config(tokenizer: AutoTokenizer, block_size: int) -> GPT2Config:
    return GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=block_size,
        n_embd=MODEL_EMBED_DIM,
        n_layer=MODEL_LAYER_COUNT,
        n_head=MODEL_HEAD_COUNT,
        resid_pdrop=MODEL_DROPOUT,
        embd_pdrop=MODEL_DROPOUT,
        attn_pdrop=MODEL_DROPOUT,
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


def load_docs_from_files(files: list[Path]) -> list[str]:
    """读取各数据文件，返回文档（每条文本一个元素）列表，保留文档边界。"""
    docs: list[str] = []
    for path in files:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if "text" not in record:
                    raise ValueError(f"{path} 中的记录缺少 text 字段。")
                docs.append(record["text"])
        else:
            docs.extend(
                line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
    return docs


def load_text_from_files(files: list[Path]) -> str:
    return "\n\n".join(load_docs_from_files(files))


def load_training_text(files: list[Path]) -> str:
    return load_text_from_files(files)


def load_training_docs(files: list[Path]) -> list[str]:
    return load_docs_from_files(files)


def tokenize_docs_with_eos(tokenizer: AutoTokenizer, docs: list[str]) -> list[int]:
    """逐文档 tokenize，并显式用 EOS 分隔文档。

    不依赖 add_special_tokens=True 是否会自动插入 EOS，确保训练与验证使用
    完全相同的文档边界规则。
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 未配置 eos_token_id，请先为 tokenizer 设置 EOS token。")

    token_stream: list[int] = []
    encoded_docs = tokenizer(docs, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    for doc_ids in encoded_docs:
        token_stream.extend(doc_ids)
        token_stream.append(tokenizer.eos_token_id)
    return token_stream


def choose_block_size(input_ids: list[int], preferred_block_size: int) -> int:
    max_allowed = len(input_ids)
    if max_allowed < 4:
        raise ValueError("文本太短，至少需要更多 token 才能训练。")
    return min(preferred_block_size, max_allowed)


def get_checkpoint_path(epoch: int) -> Path:
    return CHECKPOINT_DIR / f"epoch_{epoch:03d}"


def find_latest_checkpoint() -> Path | None:
    """返回最新保存的可恢复 checkpoint，而不是优先返回 best。

    epoch_* 与 best 都可能包含可恢复状态；按 trainer_state 中记录的 epoch
    选择实际训练进度最大的那个。best 仍只用于最终模型选择。
    """
    if not CHECKPOINT_DIR.exists():
        return None

    candidates = [path for path in CHECKPOINT_DIR.glob("epoch_*") if path.is_dir()]
    if BEST_CHECKPOINT_DIR.is_dir():
        candidates.append(BEST_CHECKPOINT_DIR)

    latest_path: Path | None = None
    latest_epoch = -1
    for path in candidates:
        state_path = path / "trainer_state.pt"
        if not state_path.exists():
            continue
        try:
            state = torch.load(state_path, map_location="cpu")
            epoch = int(state.get("epoch", -1))
        except Exception:
            continue
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path
    return latest_path


def save_checkpoint(
    checkpoint_path: Path,
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    block_size: int,
    extra_state: dict[str, object] | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    global_step: int = 0,
) -> None:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    trainer_state: dict[str, object] = {
        "epoch": epoch,
        "block_size": block_size,
        "vocab_size": len(tokenizer),
        "model_n_embd": int(model.config.n_embd),
        "model_n_layer": int(model.config.n_layer),
        "model_n_head": int(model.config.n_head),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "training_objective_version": TRAINING_OBJECTIVE_VERSION,
    }
    if scheduler is not None:
        trainer_state["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state:
        trainer_state.update(extra_state)
    torch.save(trainer_state, checkpoint_path / "trainer_state.pt")
    print(f"Checkpoint saved to: {checkpoint_path}")


def sync_best_model(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    block_size: int,
    best_eval_loss: float,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    global_step: int,
) -> None:
    save_checkpoint(
        OUTPUT_DIR,
        model,
        tokenizer,
        optimizer,
        epoch,
        block_size,
        {
            "best_eval_loss": best_eval_loss,
            "best_epoch": epoch,
            "epochs_without_improvement": 0,
            "is_best_checkpoint": True,
            "source_checkpoint_dir": str(BEST_CHECKPOINT_DIR),
            "synced_from_best_checkpoint": True,
        },
        scheduler=scheduler,
        global_step=global_step,
    )
    print(f"Best model synced to: {OUTPUT_DIR} (epoch={epoch}, eval_loss={best_eval_loss:.4f})")


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    expected_block_size: int,
    expected_vocab_size: int,
) -> tuple[GPT2LMHeadModel, dict[str, object], bool]:
    trainer_state = torch.load(checkpoint_path / "trainer_state.pt", map_location=device)
    objective_version = int(trainer_state.get("training_objective_version", 1))
    if objective_version != TRAINING_OBJECTIVE_VERSION:
        raise ValueError(
            f"Checkpoint training objective version={objective_version} 与当前 version={TRAINING_OBJECTIVE_VERSION} 不兼容。"
            "旧 checkpoint 可能使用了 double-shift labels；为避免污染，新目标将从头训练。"
        )
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


def build_dataset_from_docs(
    tokenizer: AutoTokenizer,
    docs: list[str],
    block_size: int,
) -> TextBlockDataset | None:
    input_ids = tokenize_docs_with_eos(tokenizer, docs)
    if len(input_ids) < block_size:
        return None
    return TextBlockDataset(input_ids, block_size)


def evaluate_model(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    """按有效预测 token 加权计算 loss 和 perplexity。"""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            # GPT2LMHeadModel 内部使用 labels[..., 1:] 作为预测目标。
            valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            total_nll += float(outputs.loss.item()) * valid_tokens
            total_tokens += valid_tokens
    model.train()

    if total_tokens == 0:
        raise ValueError("Evaluation dataset contains no valid prediction tokens.")
    avg_loss = total_nll / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, perplexity, total_tokens


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
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=WARMUP_RATIO,
        help="Learning rate warmup as a fraction of total training steps.",
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
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio 必须在 [0, 1) 区间内。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    # We chunk the token ids ourselves below, so the full corpus can exceed model_max_length.
    tokenizer.model_max_length = 10_000_000
    training_files = collect_training_files()
    training_docs = load_training_docs(training_files)
    # 训练与验证统一：逐文档 tokenize，并显式追加 EOS 后拍平。
    training_input_ids = tokenize_docs_with_eos(tokenizer, training_docs)
    effective_block_size = choose_block_size(training_input_ids, args.block_size)
    training_dataset = RandomWindowDataset(training_input_ids, effective_block_size)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Training files: {len(training_files)}")
    for file_path in training_files:
        print(f" - {file_path}")
    print(f"Training samples: {len(training_dataset)}, block_size: {effective_block_size}")

    eval_dir = resolve_eval_dir(args.eval_dir)
    eval_dataloader: DataLoader | None = None
    eval_dataloaders_by_source: dict[str, DataLoader] = {}
    if eval_dir is not None:
        try:
            eval_files = collect_dataset_files(eval_dir)
            eval_docs = load_docs_from_files(eval_files)
            eval_dataset = build_dataset_from_docs(tokenizer, eval_docs, effective_block_size)
            if eval_dataset is None:
                print(f"Evaluation skipped: {eval_dir} 中的文本长度不足以切出 block_size={effective_block_size} 的样本。")
            else:
                eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
                print(f"Evaluation files: {len(eval_files)}")
                for file_path in eval_files:
                    print(f" - {file_path}")
                    source_docs = load_docs_from_files([file_path])
                    source_dataset = build_dataset_from_docs(tokenizer, source_docs, effective_block_size)
                    if source_dataset is not None:
                        eval_dataloaders_by_source[file_path.stem] = DataLoader(
                            source_dataset, batch_size=args.batch_size, shuffle=False
                        )
                print(f"Evaluation samples: {len(eval_dataset)}")
        except FileNotFoundError:
            print(f"Evaluation skipped: no dataset files found in {eval_dir}")
    else:
        print("Evaluation skipped: neither ./valid nor ./test exists.")

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
                len(tokenizer),
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    can_restore_training_state = latest_checkpoint is not None and not position_embeddings_expanded
    if can_restore_training_state:
        optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
    elif position_embeddings_expanded:
        print("Skipping optimizer/scheduler state restore because position embeddings were expanded for this resume.")

    # 用完整目标 epoch 数重建同一条学习率曲线；恢复时再加载 scheduler 的 last_epoch，
    # 避免每次重启训练都重新 warmup。
    steps_per_epoch = len(dataloader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    if can_restore_training_state:
        global_step = int(trainer_state.get("global_step", start_epoch * steps_per_epoch))
        scheduler_state = trainer_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
            # load_state_dict 恢复 scheduler 计数后，同步 optimizer 当前 lr。
            for param_group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                param_group["lr"] = lr
            print(f"Scheduler restored at global_step={global_step}, lr={scheduler.get_last_lr()[0]:.8g}")
        else:
            # 兼容旧 checkpoint：直接推进到已有 global_step，不再重复 warmup。
            for _ in range(global_step):
                scheduler.step()
            print(f"Legacy checkpoint without scheduler state; advanced scheduler to global_step={global_step}.")

    if start_epoch >= args.epochs:
        print(
            f"Checkpoint has already finished {start_epoch} epochs. "
            f"Increase --epochs above {start_epoch} to continue training."
        )

    model.train()
    for epoch in range(start_epoch, args.epochs):
        training_dataset.epoch = epoch  # 每轮更新采样种子，保证窗口位置随轮次变化
        total_loss = 0.0
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, loss={avg_loss:.4f}")

        current_epoch = epoch + 1
        should_evaluate = eval_dataloader is not None and (
            current_epoch % args.eval_every == 0 or current_epoch == args.epochs
        )
        if should_evaluate:
            eval_loss, eval_ppl, eval_tokens = evaluate_model(model, eval_dataloader, device)
            print(
                f"Epoch {current_epoch}/{args.epochs}, eval_loss={eval_loss:.4f}, "
                f"eval_ppl={eval_ppl:.2f}, eval_tokens={eval_tokens}"
            )
            for source_name, source_loader in eval_dataloaders_by_source.items():
                source_loss, source_ppl, source_tokens = evaluate_model(model, source_loader, device)
                print(
                    f"  {source_name}: loss={source_loss:.4f}, "
                    f"ppl={source_ppl:.2f}, tokens={source_tokens}"
                )

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
                    scheduler=scheduler,
                    global_step=global_step,
                )
                sync_best_model(
                    model,
                    tokenizer,
                    optimizer,
                    current_epoch,
                    effective_block_size,
                    best_eval_loss,
                    scheduler,
                    global_step,
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
                scheduler=scheduler,
                global_step=global_step,
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
        shutil.copy2(BEST_CHECKPOINT_DIR / "trainer_state.pt", OUTPUT_DIR / "trainer_state.pt")
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
