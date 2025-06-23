import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadSelfAttention
from embeddings import TokenPositionalEmbedding

class CustomLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config.d_model, config.n_heads)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config.d_model, config.d_ff)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x

class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = TokenPositionalEmbedding(
            vocab_size = config.vocab_size,
            d_model = config.d_model,
            max_len = config.context_length
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
