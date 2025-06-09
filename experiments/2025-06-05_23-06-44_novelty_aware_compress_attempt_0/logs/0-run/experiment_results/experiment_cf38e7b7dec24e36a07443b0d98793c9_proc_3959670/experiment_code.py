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

    def forward(self, x, mems_x, mems_ent):
        batch, seq, dim = x.size()
        out_list, new_mems_x, new_mems_ent, effs = [], [], [], []
        eps = 1e-8
        for b in range(batch):
            x_b = x[b : b + 1]
            mem_b = None if mems_x is None else mems_x[b]
            ent_b_prev = None if mems_ent is None else mems_ent[b]
            if mem_b is None:
                k = v = x_b
            else:
                k = torch.cat([mem_b.unsqueeze(0), x_b], dim=1)
                v = k
            attn_out_b, attn_w_b = self.attn(
                x_b, k, v, need_weights=True, average_attn_weights=False
            )
            x2_b = self.norm1(x_b + attn_out_b)
            out_b = self.norm2(x2_b + self.ff(x2_b))
            aw = attn_w_b[0].mean(dim=0)
            ent_cur = -(aw * (aw + eps).log()).sum(dim=-1).detach()
            mem_new = x_b.squeeze(0).detach()
            if mem_b is None:
                mx, me = mem_new, ent_cur
            else:
                mx = torch.cat([mem_b, mem_new], dim=0)
                me = torch.cat([ent_b_prev, ent_cur], dim=0)
            if mx.size(0) > self.mem_size_base:
                thresh = me.mean() * self.threshold_factor
                idx = (me >= thresh).nonzero().squeeze(1)
                if idx.numel() > self.mem_size_base:
                    idx = torch.topk(me, self.mem_size_base).indices
                mx, me = mx[idx], me[idx]
            ent_eff = me.sum().item() / me.numel()
            out_list.append(out_b)
            new_mems_x.append(mx)
            new_mems_ent.append(me)
            effs.append(ent_eff)
        out = torch.cat(out_list, dim=0)
        return out, new_mems_x, new_mems_ent, sum(effs) / len(effs)


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

    def forward(self, x, mems_x, mems_ent):
        emb = self.embed(x)
        out_mem, mx, me, eff = self.mem_layer(emb, mems_x, mems_ent)
        logits = self.out(out_mem)
        return logits, mx, me, eff


class LMBlocksDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


# Hyperparams
block_size = 64
chunk_size = 16
mem_size_base = 32
threshold_factor = 1.0
embed_dim = 128
num_heads = 4
num_epochs = 2
batch_size = 2
max_texts = 50

random.seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
vocab_size = tokenizer.vocab_size

datasets_info = {
    "pg19": ("pg19", None),
    "arxiv": ("scientific_papers", "arxiv"),
    "wikitext": ("wikitext", "wikitext-103-raw-v1"),
}

experiment_data = {
    ds: {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ds in datasets_info
}

criterion = nn.CrossEntropyLoss()

for ds_name, (name, config) in datasets_info.items():
    print(f"\n=== Dataset: {ds_name} ===")
    ds_iter = load_dataset(name, config, split="train", streaming=True)
    texts = []
    for ex in ds_iter:
        t = ex.get("article") or ex.get("text") or ""
        if isinstance(t, dict):
            t = t.get("text", "")
        if isinstance(t, list):
            t = " ".join(t)
        if t.strip():
            texts.append(t)
        if len(texts) >= max_texts:
            break

    blocks = []
    for txt in texts:
        ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids) - block_size, block_size):
            seq = ids[i : i + block_size + 1]
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
        train_loss, train_effs = 0.0, []
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            inp, tgt = batch["input"], batch["target"]
            mems_x, mems_ent = None, None
            optimizer.zero_grad()
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mems_x, mems_ent, eff = model(ic, mems_x, mems_ent)
                loss = criterion(logits.reshape(-1, vocab_size), tc.reshape(-1))
                loss.backward()
                train_loss += loss.item()
                train_effs.append(eff)
            optimizer.step()
        avg_train_loss = train_loss / len(train_effs)
        avg_train_eff = sum(train_effs) / len(train_effs)
        experiment_data[ds_name]["losses"]["train"].append(avg_train_loss)
        experiment_data[ds_name]["metrics"]["train"].append(
            {"entropy_eff": avg_train_eff}
        )

        model.eval()
        val_loss, val_effs = 0.0, []
        val_preds, val_gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                inp, tgt = batch["input"], batch["target"]
                mems_x, mems_ent = None, None
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mems_x, mems_ent, eff = model(ic, mems_x, mems_ent)
                    loss = criterion(logits.reshape(-1, vocab_size), tc.reshape(-1))
                    val_loss += loss.item()
                    val_effs.append(eff)
                    preds = logits.argmax(dim=-1)
                    val_preds.append(preds.cpu().numpy())
                    val_gts.append(tc.cpu().numpy())
        avg_val_loss = val_loss / len(val_effs)
        avg_val_eff = sum(val_effs) / len(val_effs)
        experiment_data[ds_name]["losses"]["val"].append(avg_val_loss)
        experiment_data[ds_name]["metrics"]["val"].append({"entropy_eff": avg_val_eff})
        experiment_data[ds_name]["predictions"].append(val_preds)
        experiment_data[ds_name]["ground_truth"].append(val_gts)
        print(f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}")

    print(
        f"{ds_name} final: val_loss={avg_val_loss:.4f}, entropy_eff={avg_val_eff:.4f}"
    )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
