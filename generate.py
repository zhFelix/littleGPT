from pathlib import Path
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "littleGPT_model"
# MODEL_DIR = BASE_DIR / "checkpoints" / "epoch_300"  # 上游实验用固定 epoch 检查点，本地训练只存 best
DEFAULT_PROMPT = "黑洞是什么，它是"
MAX_NEW_TOKENS = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the current model.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory of the model to use. Defaults to ./littleGPT_model.",
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Prompt used for generation.")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use deterministic greedy decoding instead of sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = args.prompt

    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    model.eval()

    # 关键：和训练时保持一致，不自动添加 BOS / EOS
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    prompt_len = encoded["input_ids"].shape[1]

    # 动态上限：不能越过模型 n_positions=96 的上下文
    max_positions = int(getattr(model.config, "n_positions", 0) or 0)
    available_new_tokens = max_positions - prompt_len if max_positions > 0 else MAX_NEW_TOKENS
    safe_max_new_tokens = min(MAX_NEW_TOKENS, available_new_tokens)

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": safe_max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.greedy:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 0.8,
                    "top_k": 50,
                    "top_p": 0.95,
                }
            )
        output_ids = model.generate(**encoded, **generation_kwargs)

    # 只保留模型新生成的 token
    new_ids = output_ids[0, prompt_len:]

    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
