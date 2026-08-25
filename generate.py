from pathlib import Path
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "checkpoints" / "best"
DEFAULT_PROMPT = "给出一个SQL查询语句，查询所有用户的姓名和年龄"
MAX_NEW_TOKENS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the current model.")
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

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
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

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
