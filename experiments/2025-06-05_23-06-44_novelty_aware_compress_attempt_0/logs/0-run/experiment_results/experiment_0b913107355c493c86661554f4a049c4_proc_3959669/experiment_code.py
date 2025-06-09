import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Dynamic entropy‐thresholded memory layer
class MemoryTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mem_size_base, threshold_factor):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mem_size_base = mem_size_base
        self.threshold_factor = threshold_factor

    def forward(self, x, mem_x, mem_ent):
        # x: [batch=1, seq, dim]
        if mem_x is None:
            k, v = x, x
        else:
            k = torch.cat([mem_x.unsqueeze(0), x], dim=1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        # compute entropy for the single batch element
        aw = attn_w.mean(dim=1)[0]  # [seq, src]
        eps = 1e-8
        ent = -(aw * (aw + eps).log()).sum(dim=-1).detach()  # [seq]
        x_det = x.detach()[0]  # [seq, dim]
        if mem_x is None:
            mem_x_new = x_det
            mem_ent_new = ent
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent], dim=0)
        # thresholded pruning
        if mem_x_new.size(0) > self.mem_size_base:
            thresh = mem_ent_new.mean() * self.threshold_factor
            idx = (mem_ent_new >= thresh).nonzero().squeeze(1)
            if idx.numel() > self.mem_size_base:
                idx = torch.topk(mem_ent_new, self.mem_size_base).indices
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        ent_eff = mem_ent_new.sum().item() / mem_ent_new.size(0)
        return out, mem_x_new, mem_ent_new, ent_eff


class TransformerXLModel(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, mem_size_base, threshold_factor
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = MemoryTransformerLayer(
            embed_dim, num_heads, mem_size_base, threshold_factor
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mx, me, eff = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mx, me, eff


class LMBlocksDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


# Hyperparameters
block_size = 64
chunk_size = 16
mem_size_base = 32
threshold_factor = 1.0
embed_dim = 128
num_heads = 4
num_epochs = 2
batch_size = 1
max_texts = 50

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
vocab_size = tokenizer.vocab_size

datasets_info = {
    "pg19": ("pg19", None),
    "arxiv": ("scientific_papers", "arxiv"),
    "wikitext": ("wikitext", "wikitext-103-raw-v1"),
}

experiment_data = {
    ds_name: {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
    }
    for ds_name in datasets_info
}

criterion = nn.CrossEntropyLoss()

for ds_name, (name, config) in datasets_info.items():
    print(f"\n=== Dataset: {ds_name} ===")
    # streaming load, break after max_texts
    if config is None:
        ds_iter = load_dataset(name, split="train", streaming=True)
    else:
        ds_iter = load_dataset(name, config, split="train", streaming=True)
    texts = []
    for ex in ds_iter:
        if "article" in ex:
            art = ex["article"]
            if isinstance(art, dict):
                text = art.get("text", "")
                if isinstance(text, list):
                    text = " ".join(text)
            else:
                text = art
        else:
            text = ex.get("text", "")
        if text and text.strip():
            texts.append(text)
        if len(texts) >= max_texts:
            break

    # build sliding blocks
    blocks = []
    for txt in texts:
        ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids) - block_size, block_size):
            seq = ids[i : i + block_size + 1]
            if len(seq) == block_size + 1:
                inp = torch.tensor(seq[:-1], dtype=torch.long)
                tgt = torch.tensor(seq[1:], dtype=torch.long)
                blocks.append({"input": inp, "target": tgt})
        if len(blocks) >= max_texts:
            break

    random.shuffle(blocks)
    split = int(0.8 * len(blocks))
    train_blocks, val_blocks = blocks[:split], blocks[split:]
    train_loader = DataLoader(
        LMBlocksDataset(train_blocks), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(LMBlocksDataset(val_blocks), batch_size=batch_size)

    model = TransformerXLModel(
        vocab_size, embed_dim, num_heads, mem_size_base, threshold_factor
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_eff = 0.0, []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inp, tgt = batch["input"], batch["target"]
            mem_x, mem_ent = None, None
            optimizer.zero_grad()
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mem_x, mem_ent, eff = model(ic, mem_x, mem_ent)
                loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                loss.backward()
                train_loss += loss.item()
                train_eff.append(eff)
            optimizer.step()
        avg_train_loss = train_loss / len(train_eff)
        avg_train_eff = sum(train_eff) / len(train_eff)
        experiment_data[ds_name]["losses"]["train"].append(avg_train_loss)
        experiment_data[ds_name]["metrics"]["train"].append(avg_train_eff)

        model.eval()
        val_loss, val_eff = 0.0, []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                inp, tgt = batch["input"], batch["target"]
                mem_x, mem_ent = None, None
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, eff = model(ic, mem_x, mem_ent)
                    loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                    val_loss += loss.item()
                    val_eff.append(eff)
        avg_val_loss = val_loss / len(val_eff)
        avg_val_eff = sum(val_eff) / len(val_eff)
        experiment_data[ds_name]["losses"]["val"].append(avg_val_loss)
        experiment_data[ds_name]["metrics"]["val"].append(avg_val_eff)
        print(f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}")

    print(
        f"{ds_name} final: val_loss={avg_val_loss:.4f}, entropy_eff={avg_val_eff:.4f}"
    )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
