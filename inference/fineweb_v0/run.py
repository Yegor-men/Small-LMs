from inference.fineweb_v0 import architecture as arch
from inference.fineweb_v0.inference_function import inference
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("saved_models/tokenizers/fineweb/fineweb_v0")

EMBED_DIM = 768
NUM_HEADS = 12
NUM_BLOCKS = 12
MAX_SEQ_LENGTH = 1024
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

ckpt = torch.load("saved_models/models/fineweb_v0/S00000-L15.4787-E11.6345-20250410_1142.pt")
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
    temperature=1.0,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    max_gen_length=1050,
    max_context_length=MAX_SEQ_LENGTH,
)
