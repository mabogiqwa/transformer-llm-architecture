import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, context_length):
        self.context_length = context_length
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.data = [self.stoi[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y
