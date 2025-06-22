#Implementing multi-head self-attention here
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        #Linear projections for Q,K,V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        #Final linear projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape #batch, seq_len, d_model

        #Linear projections
        Q = self.q_proj(x) # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        #Split into heads
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, n_heads, T, head_dim)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        #Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5) # (B, n_heads, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores,, dim=-1) # (B, n_heads, T, T)
        attn_output = attn_weights @ V # (B, n_heads, T, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(attn_output)
