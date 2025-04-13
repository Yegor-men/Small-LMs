 
## Import statements

import torch
from torch import nn
import inspect
import logging
from transformers import PreTrainedTokenizerFast
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import logging
import matplotlib

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
# )

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = "cuda"
 
## Decoder block architecture

class CasualMaskedDecoderBlocks(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_blocks: int,
            max_seq_len: int,
            dropout: float = 0.0,
            activation_function: str = "gelu",
            ffw_network_multiplier: int = 4,
    ):
        """
        :param embed_dim: Embedding dimension of the tokens in the sequence.
        :param num_heads: Number of heads in each decoder block.
        :param num_blocks: Number of decoder blocks.
        :param dropout: Probability of dropout.
        :param max_seq_len: Maximum expected sequence length.
        :param ffw_network_multiplier: multiplier for embed_dim to get the dimensionality for the feedforward network.
        """
        super().__init__()

        assert embed_dim % num_heads == 0, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: embed_dim ({embed_dim}) is not divisible by num_heads ({num_heads})'

        assert activation_function in ("relu",
                                       "gelu"), f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: activation_function expected to be "relu"/"gelu", received "{activation_function}" instead'

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffw_network_multiplier * embed_dim,
            dropout=dropout,
            activation=activation_function,
            batch_first=True,
            norm_first=True
        )

        self.blocks = nn.ModuleList([block for _ in range(num_blocks)])

        causal = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal)

    def forward(self, tok_seq, padding_mask=None) -> torch.Tensor:
        """
        :param tok_seq: torch.Tensor of size [batch, seq_len, embed_dim] representing the pre-attended token sequence.
        :param padding_mask: Bool mask of size [batch, seq_len]; True for padding tokens.
        :return: Attended token sequence after num_blocks amount of decoder blocks.
        """
        assert tok_seq.dim() == 3, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: tok_seq tensor should be of size [batch, seq_len, embed_dim], received a tensor with {tok_seq.dim()} dimensions instead'
        batch_size, seq_len, embed_dim = tok_seq.size()
        assert seq_len <= self.max_seq_len, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: length of inputted sequence ({seq_len}) exceeds maximum expected sequence length ({self.max_seq_len})'
        assert embed_dim == self.embed_dim, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: received embed_dim ({embed_dim}) does not match the expected embed_dim ({self.embed_dim})'
        if padding_mask is not None:
            assert padding_mask.dim() == 2, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: padding_mask tensor should be of size [batch, seq_len], received a tensor with {padding_mask.dim()} dimensions instead'
            pm_batch_size, pm_seq_len = padding_mask.size()
            assert batch_size == pm_batch_size and seq_len == pm_seq_len, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: dimension mismatch between tok_seq ([{batch_size},{seq_len},{embed_dim}]) and padding_mask ([{pm_batch_size}, {pm_seq_len}])'

        casual_mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            tok_seq = block(
                src=tok_seq,
                src_mask=casual_mask,
                src_key_padding_mask=padding_mask
            )

        return tok_seq
 
## Attended token decoder

class AttendedTokenDecoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vocabulary_size: int,
    ):
        """
        :param embed_dim:
        :param vocabulary_size:
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.vocabulary_size = vocabulary_size

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=vocabulary_size)
        )

    def forward(self, att_tok_seq):
        """
        :param att_tok_seq:
        :return:
        """
        assert att_tok_seq.dim() == 3, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: att_tok_seq tensor should be of size [batch, seq_len, embed_dim], received a tensor with {att_tok_seq.dim()} dimensions instead'

        batch_size, seq_len, embed_dim = att_tok_seq.size()

        assert embed_dim == self.embed_dim, f'{self.__class__.__name__}.{inspect.currentframe().f_code.co_name}: received embed_dim ({embed_dim}) does not match expected embed_dim ({self.embed_dim})'

        token_logits = self.decoder(att_tok_seq)

        return token_logits
 
## Combined GPT model

class GPTModel(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_blocks: int,
            max_seq_len: int,
            vocab_size: int,
            tokenizer,
            dropout: float = 0.0,
            activation_function: str = "gelu",
            ffw_network_multiplier: int = 4,
    ):
        """
        :param embed_dim:
        :param num_heads:
        :param num_blocks:
        :param max_seq_len:
        :param vocab_size:
        :param tokenizer:
        :param dropout:
        :param activation_function:
        :param ffw_network_multiplier:
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.activation_function = activation_function
        self.ffw_network_multiplier = ffw_network_multiplier

        self.pad_token_id = tokenizer.pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.decoder_blocks = CasualMaskedDecoderBlocks(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            dropout=dropout,
            activation_function=activation_function,
            ffw_network_multiplier=ffw_network_multiplier,
        )

        self.decoder = AttendedTokenDecoder(
            embed_dim=embed_dim,
            vocabulary_size=vocab_size,
        )

        self.final_ln = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, tokenized_sequence):
        """
        :param tokenized_sequence:
        :return:
        """
        padding_mask = tokenized_sequence.eq(self.pad_token_id)

        batch_size, seq_length = tokenized_sequence.size()

        embedded_sequence = self.token_embedding(tokenized_sequence)

        position_ids = (
            torch.arange(seq_length, device=embedded_sequence.device)
            .unsqueeze(0)
            .expand(batch_size, seq_length)
        )
        positional_embeddings = self.positional_embedding(position_ids)

        sequence = embedded_sequence + positional_embeddings

        sequence = self.decoder_blocks(sequence, padding_mask)
        sequence = self.final_ln(sequence)
        logits = self.decoder(sequence)

        return logits
 
## Initialization

tokenizer_model_path = "../../../saved_models/tokenizers/nanogpt/nanogpt"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

EMBED_DIM = 512
NUM_HEADS = 8
NUM_BLOCKS = 18
MAX_SEQ_LENGTH = 512
VOCAB_SIZE = len(tokenizer.get_vocab())

model = GPTModel(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_size=VOCAB_SIZE,
    tokenizer=tokenizer,
    dropout=0.0,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total_params:,}")
print(f"Trainable: {total_params:,}")
 
## Data loading

import duckdb
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerFast


class ChunkedFineWebDataset(IterableDataset):
    def __init__(
            self,
            db_path: str,
            tokenizer: PreTrainedTokenizerFast,
            split: str = "train",
            val_mod: int = 0,
            num_buckets: int = 10,
            samples_per_epoch: int = 10,
            batch_size: int = 16,
            max_length: int = 512,
            stride: int = 256,
    ):
        """
        Yields lists of token IDs of length <= max_length+1.
        """
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.split = split
        self.val_mod = val_mod
        self.num_buckets = num_buckets
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride

    def __iter__(self):
        conn = duckdb.connect(self.db_path)
        if self.split == "train":
            where = f"abs(hash(id)) % {self.num_buckets} != {self.val_mod}"
        else:
            where = f"abs(hash(id)) % {self.num_buckets} = {self.val_mod}"

        batches = self.samples_per_epoch // self.batch_size
        for _ in range(batches):
            # reservoir‐sample exactly batch_size rows
            query = f"""
            SELECT text
            FROM fineweb
            TABLESAMPLE RESERVOIR({self.batch_size})
            WHERE {where}
            """
            rows = conn.execute(query).fetchall()

            # fallback if reservoir returns fewer rows (rare)
            if len(rows) < self.batch_size:
                rows = conn.execute(f"""
                  SELECT text
                  FROM fineweb
                  WHERE {where}
                  LIMIT {self.batch_size}
                """).fetchall()

            for (txt,) in rows:
                txt = txt.replace("\n", " ").strip()
                token_ids = self.tokenizer.encode(txt)
                start = 0
                while start < len(token_ids):
                    window = token_ids[start: start + self.max_length + 1]
                    if len(window) < 2:
                        break
                    yield window
                    start += (self.max_length - self.stride)

        conn.close()

    def __len__(self):
        # #windows is approximate: (samples_per_epoch * avg_tokens) / (max_length-stride)
        return (self.samples_per_epoch // self.batch_size) * self.batch_size


from torch.nn.utils.rnn import pad_sequence


class ChunkedCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # batch: List[List[int]] of length BATCH_SIZE
        input_ids = [torch.tensor(ids[:-1], dtype=torch.long) for ids in batch]
        labels = [torch.tensor(ids[1:], dtype=torch.long) for ids in batch]

        # pad to longest in batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "labels": labels}


from torch.utils.data import DataLoader

BATCH_SIZE = 16
SAMPLES_PE = 320

# 2) Datasets
train_ds = ChunkedFineWebDataset(
    db_path="../../../../New/data/fineweb/fineweb.db",
    tokenizer=tokenizer,
    split="train",
    samples_per_epoch=SAMPLES_PE,
    batch_size=BATCH_SIZE,
    max_length=MAX_SEQ_LENGTH,
    stride=256
)
val_ds = ChunkedFineWebDataset(
    db_path="../../../../New/data/fineweb/fineweb.db",
    tokenizer=tokenizer,
    split="val",
    samples_per_epoch=SAMPLES_PE,  # smaller val
    batch_size=BATCH_SIZE,
    max_length=MAX_SEQ_LENGTH,
    stride=256
)

# 3) Collator
collator = ChunkedCollator(pad_token_id=tokenizer.pad_token_id)

# 4) DataLoaders
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collator)
valid_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collator)
 
## How to save stuff

from datetime import datetime
import torch


def save_checkpoint(step, avg_loss, avg_entropy):
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": avg_loss,
        "entropy": avg_entropy,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"../../../saved_models/models/nanogpt/nanogpt-S{step + 1:05d}-L{avg_loss:.4f}-E{avg_entropy:.4f}-{ts}.pt"
    torch.save(ckpt, fname)
    print(f"Saved checkpoint to {fname}")

 
## Training loop

import torch.nn.functional as F
from torch.distributions import Categorical

NUM_EPOCHS = 10_000
LEARNING_RATE = 1e-4

scaler = GradScaler()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

num_warmup_steps = int(0.1 * NUM_EPOCHS)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=NUM_EPOCHS
)

train_losses = []
valid_losses = []
learning_rates = []
train_entropies = []
valid_entropies = []
means = []
stds = []

model.train()
optimizer.zero_grad()

"""This stuff for resuming training if I stop it"""

ckpt = torch.load("../../../saved_models/models/nanogpt/nanogpt-S03600-L12.5971-E13.9301-20250412_2003.pt")
model.load_state_dict(ckpt["model_state_dict"])
# optimizer.load_state_dict(ckpt["optimizer_state_dict"])
# scheduler.load_state_dict(ckpt["scheduler_state_dict"])
# scaler.load_state_dict(ckpt["scaler_state_dict"])
# start_step = ckpt["step"]

# for epoch in range(start_step, NUM_EPOCHS):
for epoch in range(NUM_EPOCHS):
    to_print = True if ((epoch + 1) % 1 == 0 or epoch == 0) else False
    print(f"Epoch {(epoch + 1)}/{NUM_EPOCHS} - {((epoch + 1) / NUM_EPOCHS) * 100:.3f}%",
          end=" | ") if to_print else None

    train_loss = 0
    train_entropy = 0

    micro_batches = len(train_dataloader)
    # if train_dataloader[-1]["input_ids"].size(0) != BATCH_SIZE:
    #     micro_batches -= 1

    for index, batch in enumerate(train_dataloader):
        last_lr = scheduler.get_last_lr()[0]
        learning_rates.append(last_lr)

        input_ids = batch["input_ids"].to(device)
        if input_ids.size(0) != BATCH_SIZE:
            continue
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            logits = model(input_ids)
            # print("logits mean:", logits.mean().item(), "std:", logits.std().item())
            probability_dist = F.softmax(logits[:, -1, :], dim=-1)
            entropy = torch.mean(Categorical(probs=probability_dist).entropy())
            train_entropy += entropy.detach().item()
            loss = loss_fn(
                logits.view(-1, len(tokenizer.get_vocab())),
                labels.view(-1)
            )
            loss /= micro_batches

        train_loss += loss.detach().item()
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()

    # train_loss /= micro_batches
    train_losses.append(train_loss)
    train_entropy /= micro_batches
    train_entropies.append(train_entropy)

    print(f"Train Loss: {train_loss:.5f} E {train_entropy:5f}", end=" | ") if to_print else None
    print(f"LR: {scheduler.get_last_lr()[0]:7f}", end=" | ") if to_print else None

    valid_loss = 0
    valid_entropy = 0
    skipped = 0

    for index, batch in enumerate(valid_dataloader):
        input_ids = batch["input_ids"].to(device)
        if input_ids.size(0) != BATCH_SIZE:
            skipped += 1
            continue
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            logits = model(input_ids)
            means.append(logits.mean().detach().item())
            stds.append(logits.std().detach().item())
            probability_dist = F.softmax(logits[:, -1, :], dim=-1)
            entropy = torch.mean(Categorical(probs=probability_dist).entropy())
            valid_entropy += entropy.detach().item()
            loss = loss_fn(
                logits.view(-1, len(tokenizer.get_vocab())),
                labels.view(-1)
            )

        valid_loss += loss.detach().item()

    effective_valid_dataloader_length = len(valid_dataloader) - skipped
    valid_loss /= effective_valid_dataloader_length
    valid_losses.append(valid_loss)
    valid_entropy /= effective_valid_dataloader_length
    valid_entropies.append(valid_entropy)

    print(f"Valid Loss: {valid_loss:.5f} E {valid_entropy:5f}") if to_print else None

    if ((epoch + 1) % 100 == 0) or ((epoch + 1) == NUM_EPOCHS):
        save_checkpoint(epoch, valid_loss, valid_entropy)

print("finito")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes = axes.flatten()

axes[0].plot(train_losses, label="Train losses")
axes[0].plot(valid_losses, label="Validation losses")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(train_entropies, label="Train entropies")
axes[1].plot(valid_entropies, label="Validation entropies")
axes[1].set_title("Entropy")
axes[1].legend()

axes[2].plot(learning_rates, label="LR values")
axes[2].set_title("Learning rate values over time")
axes[2].legend()

axes[3].plot(means, label="Mean logits")
axes[3].plot(stds, label="Stdevs")
axes[3].set_title("Distribution stats")
axes[3].legend()

plt.tight_layout()

plt.show()