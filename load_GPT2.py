from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

MODEL_MAX_POSITIONS = 96
MODEL_EMBED_DIM = 384
MODEL_LAYER_COUNT = 6
MODEL_HEAD_COUNT = 6

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=MODEL_MAX_POSITIONS,
    n_embd=MODEL_EMBED_DIM,
    n_layer=MODEL_LAYER_COUNT,
    n_head=MODEL_HEAD_COUNT,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
print(model.num_parameters())
