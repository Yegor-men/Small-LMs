import architecture as arch
from inference_function import inference
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("new/minigpt/saved/tokenizer")

EMBED_DIM = 512
NUM_HEADS = 8
NUM_BLOCKS = 8
MAX_SEQ_LENGTH = 512
VOCAB_SIZE = len(tokenizer.get_vocab())

model = arch.GPTModel(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_size=VOCAB_SIZE,
    tokenizer=tokenizer,
    dropout=0.0,
).to("cuda")

ckpt = torch.load("new/minigpt/saved/model/minigpt-S06000-L5.1952-E6.6401-20250414_1007.pt")
model.load_state_dict(ckpt["model_state_dict"])
# optimizer.load_state_dict(ckpt["optimizer_state_dict"])
# scheduler.load_state_dict(ckpt["scheduler_state_dict"])
# scaler.load_state_dict(ckpt["scaler_state_dict"])
# start_step = ckpt["step"]

model.eval()

input_text = "Once upon a time, there was a fantastic cat that"

inference(
    model,
    tokenizer,
    input_text,
    sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    max_gen_length=700,
    max_context_length=MAX_SEQ_LENGTH,
)
