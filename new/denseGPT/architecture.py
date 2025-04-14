import torch
from torch import nn


class DenseFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        growth_rate: int = 32,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(
                in_features=2 * embed_dim, out_features=2 * embed_dim + growth_rate
            ),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(
                in_features=4 * embed_dim, out_features=4 * embed_dim + 2 * growth_rate
            ),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(
                in_features=8 * embed_dim + 3 * growth_rate, out_features=embed_dim
            ),
            nn.LeakyReLU(),
        )

    def forward(self, unattended, attended):
        l1_in = torch.cat([unattended, attended], dim=-1)  # e+e=2e
        l1_out = self.layer1(l1_in)  # 2e+g
        l2_in = torch.cat([l1_in, l1_out], dim=-1)  # 2e+(2e+g)=4e+g
        l2_out = self.layer2(l2_in)  # 4e+2g
        l3_in = torch.cat([l2_in, l2_out], dim=-1)  # (4e+g)+(4e+2g)=8e+3g
        l3_out = self.layer3(l3_in)

        return l3_out


class DenseTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        assert (
            embed_dim % num_heads == 0
        ), f"DenseTransformer: embed_dim ({embed_dim}) not divisible by num_heads ({num_heads})"

        self.q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.k = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.v = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.pre_dense_norm = nn.LayerNorm(embed_dim)
        self.post_dense_norm = nn.LayerNorm(embed_dim)

        self.dense_ffn = DenseFFN(
            embed_dim=embed_dim,
            growth_rate=32,
        )

    def forward(self, token_seq, padding_mask=None):
        assert (
            token_seq.size(-1) == self.embed_dim
        ), f"DenseTransformer: token_seq.size(-1) is {token_seq.size(-1)} and not {self.embed_dim}"

        q, k, v = self.q(token_seq), self.k(token_seq), self.v(token_seq)

        seq_len = token_seq.size(1)
        casual_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        ).to(token_seq.device)

        attended_sequence, _ = self.mha(
            query=q,
            key=k,
            value=v,
            attn_mask=casual_mask,
            key_padding_mask=~padding_mask if padding_mask is not None else None,
        )

        temp1 = token_seq + attended_sequence
        norm_temp = self.pre_dense_norm(temp1)

        dense_out = self.dense_ffn(token_seq, attended_sequence)

        temp2 = norm_temp + dense_out + token_seq
        out_sequence = self.post_dense_norm(temp2)

        return out_sequence


class DenseCompressor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.layer1 = nn.Linear(in_features=2 * embed_dim, out_features=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pre_attended, attended):
        assert (
            pre_attended.size(-1) == attended.size(-1) == self.embed_dim
        ), f"DenseCompressor: pre_attended and/or attended are of the wrong size"

        combined = torch.cat([pre_attended, attended], dim=-1)
        compressed = self.layer1(combined)
        normalized = self.norm(compressed)

        return normalized


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        precompute_seq_len: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        
        self.base = base

        assert embed_dim % 2 == 0, f"TokenEmbedding: embed_dim must be even for RoPE"

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )

        frequencies = 1.0 / (
            base ** (torch.arange(0, embed_dim, 2).float() / embed_dim)
        )

        pos = torch.arange(precompute_seq_len)
        frequencies = torch.outer(pos, frequencies)

        self.register_buffer("cos_cached", torch.cos(frequencies))
        self.register_buffer("sin_cached", torch.sin(frequencies))

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        seq_len = x.size(1)

        if seq_len <= self.cos_cached.size(0):
            cos = self.cos_cached[:seq_len, :]
            sin = self.sin_cached[:seq_len, :]
        else:
            frequencies = 1.0 / (
                self.base
                ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)
            )
            pos = torch.arange(seq_len, device=x.device)
            frequencies = torch.outer(pos, frequencies)
            cos = torch.cos(frequencies)
            sin = torch.sin(frequencies)

        x_rot = x[..., ::2]  # [batch, seq_len, dim/2]
        x_pass = x[..., 1::2]  # [batch, seq_len, dim/2]

        x_rotated = torch.stack(
            [
                x_rot * cos - x_pass * sin,  # Real component
                x_rot * sin + x_pass * cos,  # Imaginary component
            ],
            dim=-1,
        ).flatten(
            -2
        )  # [batch, seq_len, dim]

        return x_rotated


class DenseDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.layer1 = nn.Linear(in_features=embed_dim, out_features=vocab_size)

    def forward(self, attended_token_sequence):
        return self.layer1(attended_token_sequence)


class DenseGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        padding_token_id: int,
    ):
        super().__init__()

        self.padding_token_id = padding_token_id

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            precompute_seq_len=2048,
            base=10000,
        )

        dt = DenseTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        dc = DenseCompressor(
            embed_dim=embed_dim,
        )

        self.DTs = nn.ModuleList(
            [
                DenseTransformer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        self.DCs = nn.ModuleList(
            [
                DenseCompressor(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        self.decoder = DenseDecoder(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
        )

    def forward(self, token_ids):

        padding_mask = token_ids != self.padding_token_id

        token_sequence = self.embedding(token_ids)

        for dt_block, dc_block in zip(self.DTs, self.DCs):
            attended_seq = dt_block(token_sequence, padding_mask)
            token_sequence = dc_block(token_sequence, attended_seq)

        logits = self.decoder(token_sequence)

        return logits
