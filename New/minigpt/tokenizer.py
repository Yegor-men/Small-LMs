## Import statements

from datasets import load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast
## Data loading

import duckdb
from typing import Iterator, List, Optional


def get_training_corpus(
        db_path: str = "../../data/fineweb/fineweb.db",
        batch_size: int = 1_000,
        max_entries: Optional[int] = None
) -> Iterator[List[str]]:
    """
    Stream batches of 'text' out of the fineweb DuckDB table.

    Args:
        db_path:      Path to your .db file.
        batch_size:   How many rows to pull per batch.
        max_entries:  If set, stops after yielding this many total rows.

    Yields:
        Lists of `batch_size` text strings (last batch may be smaller).
    """
    conn = duckdb.connect(db_path)
    # Open a cursor on the text column only
    cur = conn.cursor().execute("SELECT text FROM fineweb")
    total_yielded = 0

    while True:
        # fetchmany is efficient and avoids OFFSET
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        texts = [r[0] for r in rows]

        # If the caller only wants the first N entries, truncate and stop
        if max_entries is not None:
            remaining = max_entries - total_yielded
            if remaining <= 0:
                break
            if len(texts) > remaining:
                texts = texts[:remaining]
                yield texts
                break

        yield texts
        total_yielded += len(texts)

    conn.close()


## Initialization/training

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
special_tokens = ["<eos>", "<pad>", "<unk>", "<model>", "</model>", "<user>", "</user>", "<system>", "</system>",
                  "<think>", "</think>", ]
total = len(special_tokens) + 50_000
trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train_from_iterator(get_training_corpus(max_entries=100000), trainer=trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

## Testing

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
print(encoding.ids)
print(tokenizer.decode(encoding.ids))
print(len(tokenizer.get_vocab()))
print(tokenizer.get_vocab_size())

## Saving

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<pad>",
    eos_token="<eos>",
    unk_token="<unk>",
    # cls_token="[CLS]",
    # sep_token="[SEP]",
    # mask_token="[MASK]",
)

wrapped_tokenizer.save_pretrained("../../../saved_models/tokenizers/nanogpt/nanogpt")

tok = AutoTokenizer.from_pretrained("../../../saved_models/tokenizers/nanogpt/nanogpt")

tokens = tok.tokenize("Test number 2.!@#$%^&*() Yohoho skibidi biden <think></think><eos>")

print(tokens)
ids = tok.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tok.decode(ids)
print(decoded_string)
print(len(tok.get_vocab()))
