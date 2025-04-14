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