import torch
import torch.nn as nn

class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = LearnedPositionalEncoding(max_len, d_model)

    def forward(self, x):
        token_embeddings = self.token_embed(x)
        return self.pos_embed(token_embeddings)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pos_embed(pos)
