import torch
from attention import MultiHeadSelfAttention
from utils import generate_causal_mask

batch_size = 2
seq_len = 4
d_model = 128
n_heads = 4

# Creating dummy input
x = torch.randn(batch_size, seq_len, d_model) # (B, T, d_model)

#Initialize attention module
attention = MultiHeadSelfAttention(d_model, n_heads)


#Forward pass
output = attention(x)

print("Input shape: ", x.shape)
print("Output shape: ", output.shape)
