from pathlib import Path
import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_model"
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
        "--prompt",
        action="append",
        dest="prompts",
        help="Add a prompt to test. You can pass this option multiple times.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=40, help="Maximum tokens to generate per prompt.")
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


def compute_prompt_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
) -> tuple[float, float]:
    encoded = tokenizer(prompt, return_tensors="pt")
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
) -> tuple[float, float]:
    losses = []
    for line in lines:
        loss, _ = compute_prompt_loss(model, tokenizer, line["text"], device)
        losses.append(loss)

    average_loss = sum(losses) / len(losses)
    average_perplexity = math.exp(average_loss)
    return average_loss, average_perplexity


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    max_positions = int(getattr(model.config, "n_positions", 0) or 0)
    prompt_length = int(encoded["input_ids"].shape[1])
    available_new_tokens = max_positions - prompt_length if max_positions > 0 else max_new_tokens

    if available_new_tokens <= 0:
        return "[skipped generation: prompt length already reaches the model context limit]"

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
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
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
    print(f"Model dir: {MODEL_DIR}")
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
        overall_losses = []
        total_lines = 0
        for file_name, lines in test_sets.items():
            dataset_loss, dataset_perplexity = evaluate_dataset(model, tokenizer, lines, device)
            overall_losses.extend(compute_prompt_loss(model, tokenizer, line["text"], device)[0] for line in lines)
            total_lines += len(lines)
            print(
                f"{file_name}: lines={len(lines)}, avg_loss={dataset_loss:.4f}, avg_perplexity={dataset_perplexity:.2f}"
            )

        if total_lines:
            overall_loss = sum(overall_losses) / len(overall_losses)
            print(
                f"overall: lines={total_lines}, avg_loss={overall_loss:.4f}, avg_perplexity={math.exp(overall_loss):.2f}"
            )
        print("-" * 80)
    else:
        print("No test dataset files were loaded, so the script is using prompt-based smoke tests.")
        print("-" * 80)

    for index, prompt in enumerate(prompts, start=1):
        loss, perplexity = compute_prompt_loss(model, tokenizer, prompt, device)
        generated = generate_text(
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
        print("-" * 80)


if __name__ == "__main__":
    main()
