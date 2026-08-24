from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_embd=512,
    n_layer=8,
    n_head=8,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(config)
print(model.num_parameters())
