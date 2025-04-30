import architecture

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from torch.amp import autocast, GradScaler

from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


device = "cuda"

tokenizer_model_path = "new/denseGPT/saved/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)

model = architecture.DenseGPT(
    vocab_size=len(tokenizer.get_vocab()),
    embed_dim=384,
    num_heads=6,
    num_blocks=6,
    padding_token_id=tokenizer.pad_token_id,
).to(device)

r_value = 422334706
BETA_VALUE = 50
total_tokens = int(r_value * BETA_VALUE)

SEQ_LEN = 1024
BATCH_SIZE = 16
ALPHA_VALUE = 0.01
tokens_in_update_step = int(ALPHA_VALUE * r_value)
num_minibatches = int(round(tokens_in_update_step / (SEQ_LEN * BATCH_SIZE)))
effective_tokens_in_update_step = int(num_minibatches * BATCH_SIZE * SEQ_LEN)

NUM_UPDATE_STEPS = int(round(total_tokens / effective_tokens_in_update_step))

# num_batches_in_dataloader = 30
# NUM_UPDATE_STEPS = 10_000
start_step = 0
LR = 1e-4

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * NUM_UPDATE_STEPS),
    num_training_steps=NUM_UPDATE_STEPS,
)


import torch
from torch.utils.data import Dataset, DataLoader
import duckdb
import numpy as np
from typing import List, Tuple, Dict


class ChunkedTextDataset(Dataset):
    def __init__(
        self,
        db_path: str,
        tokenizer,
        split_type: str,
        seq_len: int,
        batch_size: int,
        num_minibatches: int,
        memory_limit_gb: int = 16,
    ):
        super().__init__()
        self.db_path = db_path
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.memory_limit_gb = memory_limit_gb

        with duckdb.connect(db_path, read_only=True) as conn:
            conn.execute(f"SET memory_limit='{memory_limit_gb}GB'")
            total_rows = conn.execute("SELECT COUNT(*) FROM fineweb").fetchone()[0]

        train_start, train_end = 0, int(total_rows * 0.8)
        val_start, val_end = int(total_rows * 0.8), int(total_rows * 0.9)
        test_start, test_end = int(total_rows * 0.9), total_rows

        if split_type == "train":
            self.start_index = train_start
            self.end_index = train_end
            self.should_cycle = True
        elif split_type == "val":
            self.start_index = val_start
            self.end_index = val_end
            self.should_cycle = False
        elif split_type == "test":
            self.start_index = test_start
            self.end_index = test_end
            self.should_cycle = False

        self.current_index = self.start_index
        self.conn = None
        self.current_batch_tokens = []

        def _get_connection(self):
            if self.conn is None:
                self.conn = duckdb.connect(self.db_path, read_only=True)
                conn.execute(f"SET memory_limit='{self.memory_limit_gb}GB'")
            return self.conn

        def load_next_batch(self):
            self.current_batch_tokens = []

            with self._get_connection() as conn:
                while len(self.current_batch_tokens) < self.seq_len * self.batch_size:

                    query = f"""
                    SELECT text FROM fineweb
                    where rowid = {self.current_index}
                    LIMIT 1
                    """
                    result = conn.execute(query).fetchone()

                    if result:
                        text = result[0]
                        tokens = self.tokenizer.encode(
                            text,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.seq_length,
                        )

                    self.current_index += 1
                    if self.current_index >= self.end_index:
                        if self.should_cycle:
                            self.current_index = self.start_index
                        else:
                            break

        self.current_batch_tokens = self.current_batch_tokens[
            : self.batch_size * self.seq_len
        ]

    def __len__(self):
        return self.batch_size * self.num_minibatches

    def __getitem__(self, index):
        batch_index = index // self.batch_size
        item_index = index % self.batch_size
        if batch_index >= self.num_minibatches:
            raise StopIteration
        if len(self.current_batch_tokens) < self.seq_len:
            self.load_next_batch()
        start_index = item_index * self.seq_len
        sequence = torch.tensor(
            self.current_batch_tokens[start_index : start_index + self.seq_len]
        )
        if len(sequence) < self.seq_len:
            padding = torch.full(
                (self.seq_len - len(sequence),), self.tokenizer.pad_token_id
            )
            sequence = torch.cat([sequence, padding])
        input_ids = sequence
        labels = torch.roll(sequence, shifts=-1)
        labels[-1] = self.tokenizer.eos_token_id

        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(batch) < BATCH_SIZE:
        return None  # Skip incomplete batches
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


# Create datasets
train_dataset = ChunkedTextDataset(
    db_path="data/fineweb/fineweb.db",
    tokenizer=tokenizer,
    split_type="train",
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    num_minibatches=num_minibatches,
)

val_dataset = ChunkedTextDataset(
    db_path="data/fineweb/fineweb.db",
    tokenizer=tokenizer,
    split_type="val",
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    num_minibatches=num_minibatches,
)

test_dataset = ChunkedTextDataset(
    db_path="data/fineweb/fineweb.db",
    tokenizer=tokenizer,
    split_type="test",
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    num_minibatches=num_minibatches,
)


# Create dataloaders
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    worker_init_fn=worker_init_fn,
    persistent_workers=False,
    prefetch_factor=2,
    pin_memory=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    worker_init_fn=worker_init_fn,
    persistent_workers=False,
    prefetch_factor=2,
    pin_memory=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    worker_init_fn=worker_init_fn,
    persistent_workers=False,
    prefetch_factor=2,
    pin_memory=True,
)

from datetime import datetime


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
    fname = f"new/denseGPT/saved/model-S{step + 1:05d}-L{avg_loss:.4f}-E{avg_entropy:.4f}-{ts}.pt"
    torch.save(ckpt, fname)
    print(f"Saved checkpoint to {fname}")


def plot_training_stats(
    train_losses,
    validation_losses,
    train_entropies,
    validation_entropies,
    learning_rates,
    update_step,
):
    plt.close("all")
    plt.clf()  # Clear the current figure
    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    axes = axes.flatten()

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(validation_losses, label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(train_entropies, label="Train")
    axes[1].plot(validation_entropies, label="Validation")
    axes[1].set_title("Entropy")
    axes[1].legend()

    plt.tight_layout()

    # Create plots directory if it doesn't exist
    plots_dir = Path("new/denseGPT/plots")
    plots_dir.mkdir(exist_ok=True)

    # Save the plot
    plt.savefig(plots_dir / f"training_stats_step_{update_step}.png")
    plt.close()


train_losses, test_losses, validation_losses = [], [], []
train_entropies, test_entropies, validation_entropies = [], [], []
train_means, test_means, validation_means = [], [], []
train_stdevs, test_stdevs, validation_stdevs = [], [], []
learning_rates = []

model.train()
optimizer.zero_grad()


# THIS STUFF IS FOR CONTINUING TRAINING GIVEN A SAVED MODEL
# ckpt = torch.load("new/denseGPT/saved/model")
# model.load_state_dict(ckpt["model_state_dict"])
# optimizer.load_state_dict(ckpt["optimizer_state_dict"])
# scheduler.load_state_dict(ckpt["scheduler_state_dict"])
# scaler.load_state_dict(ckpt["scaler_state_dict"])
# start_step = ckpt["step"]


for update_step in range(start_step, NUM_UPDATE_STEPS):
    print(
        f"U{update_step+1:,}/{NUM_UPDATE_STEPS:,} - {(update_step+1)/NUM_UPDATE_STEPS:.2f}%"
    )

    train_loss, train_entropy, train_mean, train_stdev = 0, 0, 0, 0
    model.train()
    for index, batch in tqdm(enumerate(train_dataloader), desc="Train", colour="red"):
        last_lr = scheduler.get_last_lr()[0]
        learning_rates.append(last_lr)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            logits = model(input_ids)
            # logits_mean = logits.mean().item()
            # logits_stdev = logits.std.item()
            last_token_probability_dist = F.softmax(logits[:, -1, :], dim=-1)
            entropy = torch.mean(
                Categorical(probs=last_token_probability_dist).entropy()
            )
            train_entropy += entropy.detach().item()
            loss = loss_fn(logits.view(-1, len(tokenizer.get_vocab())), labels.view(-1))
            loss /= num_minibatches
            train_loss += loss.detach().item()

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()

    train_entropy /= num_minibatches
    train_losses.append(train_loss)
    train_entropies.append(train_entropy)

    valid_loss, valid_entropy, valid_mean, valid_stdev = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for index, batch in tqdm(
            enumerate(val_dataloader), desc="Validate", colour="yellow"
        ):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast("cuda"):
                logits = model(input_ids)
                # logits_mean = logits.mean().item()
                # logits_stdev = logits.std.item()
                last_token_probability_dist = F.softmax(logits[:, -1, :], dim=-1)
                entropy = torch.mean(
                    Categorical(probs=last_token_probability_dist).entropy()
                )
                valid_entropy += entropy.detach().item()
                loss = loss_fn(
                    logits.view(-1, len(tokenizer.get_vocab())), labels.view(-1)
                )
                valid_loss += loss.detach().item()

        valid_loss /= num_minibatches
        valid_entropy /= num_minibatches
        validation_losses.append(valid_loss)
        validation_entropies.append(valid_entropy)

    if ((update_step + 1) % 100 == 0) or ((update_step + 1) == NUM_UPDATE_STEPS):
        save_checkpoint(update_step, valid_loss, valid_entropy)

    if ((update_step + 1) % 1 == 0) or ((update_step + 1) == NUM_UPDATE_STEPS):
        plot_training_stats(
            train_losses,
            validation_losses,
            train_entropies,
            validation_entropies,
            learning_rates,
            update_step,
        )


test_loss, test_entropy, test_mean, test_stdev = 0, 0, 0, 0
model.eval()
with torch.no_grad():
    for index, batch in tqdm(enumerate(test_dataloader), desc="Test", colour="green"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda"):
            logits = model(input_ids)
            # logits_mean = logits.mean().item()
            # logits_stdev = logits.std.item()
            last_token_probability_dist = F.softmax(logits[:, -1, :], dim=-1)
            entropy = torch.mean(
                Categorical(probs=last_token_probability_dist).entropy()
            )
            test_entropies.append(entropy.detach().item())
            loss = loss_fn(logits.view(-1, len(tokenizer.get_vocab())), labels.view(-1))
            test_losses.append(loss.detach().item())
