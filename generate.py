from pathlib import Path
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
# MODEL_DIR = BASE_DIR / "checkpoints" / "best"
MODEL_DIR = BASE_DIR / "checkpoints" / "epoch_300"
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
    parser.add_argument(
        "--show-special-tokens",
        action="store_true",
        help="Show special tokens in the generated continuation for debugging.",
    )
    return parser.parse_args()


def truncate_at_first_eos(token_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Return only tokens before the first EOS token."""
    eos_positions = (token_ids == eos_token_id).nonzero(as_tuple=False)
    if eos_positions.numel() == 0:
        return token_ids
    first_eos = int(eos_positions[0].item())
    return token_ids[:first_eos]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = args.prompt

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; generation cannot stop reliably at document boundaries.")

    # Training tokenization used add_special_tokens=False, so inference should match it.
    # In particular, do not prepend a BOS token that the training stream did not contain.
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": pad_token_id,
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

    # model.generate returns: [prompt tokens] + [newly generated tokens].
    # Only decode the continuation, not the prompt itself.
    new_token_ids = output_ids[0, prompt_length:]

    # Defensive second layer: even if generate() does not stop for some config reason,
    # never expose text after the first EOS, because training uses EOS as a document boundary.
    visible_token_ids = truncate_at_first_eos(new_token_ids, tokenizer.eos_token_id)

    generated_text = tokenizer.decode(
        visible_token_ids,
        skip_special_tokens=not args.show_special_tokens,
    )

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    if args.show_special_tokens:
        # Show the raw continuation as an additional diagnostic view. This can reveal
        # whether generate() itself emitted EOS before any later tokens.
        raw_generated = tokenizer.decode(new_token_ids, skip_special_tokens=False)
        print(f"Raw generated: {raw_generated}")
        print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
        print(f"model.config.eos_token_id: {model.config.eos_token_id}")
        print(f"model.generation_config.eos_token_id: {model.generation_config.eos_token_id}")


if __name__ == "__main__":
    main()
