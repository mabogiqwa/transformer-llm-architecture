import torch

def generate_causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0) # (1, 1, size, size)
