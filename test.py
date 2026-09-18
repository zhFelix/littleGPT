from pathlib import Path
import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "checkpoints" / "best"
TEST_DIR = BASE_DIR / "test"
TEST_FILES = ["chinese", "english"]
DEFAULT_PROMPTS = [
    "中国的首都是",
    "量子力学研究",
    "Machine learning is",
    "A useful scientific theory",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a final smoke test for the trained GPT-2 model.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory of the model to test. Defaults to ./checkpoints/best.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Add a prompt to test. You can pass this option multiple times.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximum tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling value.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling value.")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=2,
        help="Number of sample prompts to take from each test file when no custom prompts are provided.",
    )
    return parser.parse_args()


def get_block_size(model: AutoModelForCausalLM) -> int:
    return int(getattr(model.config, "n_positions", 0) or 0)


def compute_prompt_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
) -> tuple[float, float]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    # 防止 prompt 超过模型 context 导致越界
    max_positions = get_block_size(model)
    if max_positions > 0:
        encoded["input_ids"] = encoded["input_ids"][:, :max_positions]
        if "attention_mask" in encoded:
            encoded["attention_mask"] = encoded["attention_mask"][:, :max_positions]
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, labels=encoded["input_ids"])

    loss = float(outputs.loss.item())
    perplexity = math.exp(loss)
    return loss, perplexity


def load_test_lines() -> dict[str, list[dict[str, str]]]:
    datasets: dict[str, list[dict[str, str]]] = {}
    if not TEST_DIR.exists():
        return datasets
    for dataset_name in TEST_FILES:
        jsonl_path = TEST_DIR / f"{dataset_name}.jsonl"
        txt_path = TEST_DIR / f"{dataset_name}.txt"

        if jsonl_path.exists():
            records = []
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if "text" not in record:
                    raise ValueError(f"{jsonl_path} 中的记录缺少 text 字段。")
                records.append(record)
            datasets[jsonl_path.name] = records
            continue

        if txt_path.exists():
            records = [{"text": line.strip()} for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            datasets[txt_path.name] = records
    return datasets


def evaluate_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lines: list[dict[str, str]],
    device: torch.device,
) -> tuple[float, float, int, int]:
    """token-weighted loss / PPL 评估。

    长文本按 block_size 切块，每个 token 累计 NLL，最后
    总 NLL / 总有效 token 再 exp()，避免短文本被过度放大，
    同时防止超过模型 context。
    """
    block_size = get_block_size(model)
    if block_size <= 0:
        block_size = 96  # 兜底：与训练 BLOCK_SIZE 一致
    total_nll = 0.0
    total_tokens = 0

    for line in lines:
        ids = tokenizer(line["text"], add_special_tokens=False, return_attention_mask=False)["input_ids"]
        if len(ids) < 2:
            continue
        for start in range(0, len(ids), block_size):
            block = ids[start : start + block_size]
            if len(block) < 2:
                continue
            input_ids = torch.tensor([block], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
            # GPT2LMHeadModel 内部 shift 后有效预测数为 len(block) - 1
            valid_tokens = len(block) - 1
            total_nll += float(outputs.loss.item()) * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        return 0.0, 0.0, len(lines), 0
    average_loss = total_nll / total_tokens
    return average_loss, math.exp(average_loss), len(lines), total_tokens


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: int,
) -> tuple[str, bool]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    max_positions = get_block_size(model)
    prompt_length = int(encoded["input_ids"].shape[1])
    available_new_tokens = max_positions - prompt_length if max_positions > 0 else max_new_tokens

    if available_new_tokens <= 0:
        return "[skipped generation: prompt length already reaches the model context limit]", False

    safe_max_new_tokens = min(max_new_tokens, available_new_tokens)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=safe_max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成部分，方便判断模型到底生成了什么
    new_ids = output_ids[0, prompt_length:]
    ended_with_eos = len(new_ids) > 0 and int(new_ids[-1].item()) == int(tokenizer.eos_token_id)
    return tokenizer.decode(new_ids, skip_special_tokens=True), ended_with_eos


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    model.eval()
    test_sets = load_test_lines()

    prompts = args.prompts
    if not prompts:
        prompts = []
        if test_sets:
            for records in test_sets.values():
                prompts.extend(record["text"] for record in records[: args.sample_count])
        else:
            prompts = DEFAULT_PROMPTS

    print(f"Using device: {device}")
    print(f"Model dir: {model_dir}")
    print(f"Test dir: {TEST_DIR}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"BOS token id: {tokenizer.bos_token_id}")
    print(f"EOS token id: {tokenizer.eos_token_id}")
    print(f"Model max positions: {getattr(model.config, 'n_positions', 'unknown')}")
    print(f"Test dir exists: {TEST_DIR.exists()}")
    print(f"Prompt count: {len(prompts)}")
    print("-" * 80)

    if test_sets:
        print("[Dataset Evaluation]")
        overall_nll = 0.0
        overall_tokens = 0
        total_lines = 0
        for file_name, lines in test_sets.items():
            dataset_loss, dataset_perplexity, num_lines, num_tokens = evaluate_dataset(
                model, tokenizer, lines, device
            )
            overall_nll += dataset_loss * num_tokens
            overall_tokens += num_tokens
            total_lines += num_lines
            print(
                f"{file_name}: lines={num_lines}, tokens={num_tokens}, "
                f"avg_loss={dataset_loss:.4f}, avg_perplexity={dataset_perplexity:.2f}"
            )

        if overall_tokens:
            overall_loss = overall_nll / overall_tokens
            print(
                f"overall: lines={total_lines}, tokens={overall_tokens}, "
                f"avg_loss={overall_loss:.4f}, avg_perplexity={math.exp(overall_loss):.2f}"
            )
        print("-" * 80)
    else:
        print("No test dataset files were loaded, so the script is using prompt-based smoke tests.")
        print("-" * 80)

    for index, prompt in enumerate(prompts, start=1):
        loss, perplexity = compute_prompt_loss(model, tokenizer, prompt, device)
        generated, ended_with_eos = generate_text(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )

        print(f"[Test {index}]")
        print(f"Prompt: {prompt}")
        print(f"Prompt loss: {loss:.4f}")
        print(f"Prompt perplexity: {perplexity:.2f}")
        print(f"Generated: {generated}")
        print(f"Ended with EOS: {ended_with_eos}")
        print("-" * 80)


if __name__ == "__main__":
    main()
