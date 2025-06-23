import torch
from model import TinyTransformer
from config import Config

cfg = Config()

#Dummy batch of token IDs
B, T = 2, cfg.context_length #batch size, sequence length
dummy_input = torch.randint(0, cfg.vocab_size, (B, T)) # (B, T)

#Initializing model
model = TinyTransformer(cfg)

#Forward pass
logits = model(dummy_input) #Output shape: (B, T, vocab_size)

#Print shapes
print("Input shape:", dummy_input.shape)
print("Logits shape:", logits.shape)
