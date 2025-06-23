import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TinyTransformer
from data import CharDataset
from config import Config

cfg = Config()

with open("data/dataset.txt","r",encoding="utf-8") as f:
    text = f.read()

dataset = CharDataset(text, cfg.context_length)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

model = TinyTransformer(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(cfg.num_epochs):
    total_loss = 0
    model.train()

    for batch in dataloader:
        inputs, targets = [b.to(device) for b in batch] #(B, T)
        logits = model(inputs) #(B,T, vocab_size)

        #Flatten for loss: (B*T, vocab_size), (B*T)
        loss = criterion(logits.view(-1, cfg.vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{cfg.num_epochs} | Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "tiny_transformer.pt")
