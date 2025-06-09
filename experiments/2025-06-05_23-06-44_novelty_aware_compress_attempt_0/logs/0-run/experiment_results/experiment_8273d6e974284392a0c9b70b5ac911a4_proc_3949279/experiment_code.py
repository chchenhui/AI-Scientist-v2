import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic dataset
class RandomSeqDataset(Dataset):
    def __init__(self, num_seqs, total_len, vocab_size):
        self.data = torch.randint(
            1, vocab_size, (num_seqs, total_len), dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return {"input": seq[:-1], "target": seq[1:]}


vocab_size = 50
seq_total = 51  # input 50, target 50
train_ds = RandomSeqDataset(200, seq_total, vocab_size)
val_ds = RandomSeqDataset(50, seq_total, vocab_size)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)


# Model with entropy‐aware memory
class MemoryTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mem_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mem_size = mem_size

    def forward(self, x, mem_x, mem_ent):
        if mem_x is None:
            k = v = x
        else:
            k = torch.cat([mem_x.unsqueeze(0), x], dim=1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        aw = attn_w.mean(dim=1)[0]  # (tgt_len, src_len)
        eps = 1e-10
        ent = -(aw * (aw + eps).log()).sum(dim=-1).detach()  # (tgt_len,)
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new = x_det
            mem_ent_new = ent
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent], dim=0)
        total_ent = mem_ent_new.sum().item() + eps
        if mem_x_new.size(0) > self.mem_size:
            idx = torch.topk(mem_ent_new, self.mem_size).indices
            kept_ent = mem_ent_new[idx].sum().item()
            ratio = kept_ent / total_ent
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


class TransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = MemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


# Hyperparameter tuning over learning rates
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
experiment_data = {"learning_rate": {}}

num_epochs = 3
chunk_size = 10

for lr in learning_rates:
    key = f"{lr}"
    print(f"\n=== Tuning learning rate: {key} ===")
    # initialize model, optimizer, criterion
    model = TransformerXLModel(vocab_size, embed_dim=64, num_heads=2, mem_size=20).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # prepare storage
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_ratios = 0.0, []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inp, tgt = batch["input"], batch["target"]
            mem_x = mem_ent = None
            optimizer.zero_grad()
            loss_accum = 0.0
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                loss = criterion(logits.reshape(-1, vocab_size), tc.reshape(-1))
                loss_accum += loss
                train_ratios.append(ratio)
            loss_accum.backward()
            optimizer.step()
            train_loss += loss_accum.item() / (inp.size(1) / chunk_size)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ratio = sum(train_ratios) / len(train_ratios)
        exp["losses"]["train"].append(avg_train_loss)
        exp["metrics"]["train"].append(avg_train_ratio)
        # validation loop
        model.eval()
        val_loss, val_ratios = 0.0, []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                inp, tgt = batch["input"], batch["target"]
                mem_x = mem_ent = None
                loss_acc = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    loss_acc += criterion(
                        logits.reshape(-1, vocab_size), tc.reshape(-1)
                    )
                    val_ratios.append(ratio)
                val_loss += loss_acc.item() / (inp.size(1) / chunk_size)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ratio = sum(val_ratios) / len(val_ratios)
        exp["losses"]["val"].append(avg_val_loss)
        exp["metrics"]["val"].append(avg_val_ratio)
        print(f"LR {key} Epoch {epoch} val_loss={avg_val_loss:.4f}")
    # generation on first val sequence
    model.eval()
    with torch.no_grad():
        sample = val_ds[0]["input"].unsqueeze(0).to(device)
        target = val_ds[0]["target"].tolist()
        mem_x = mem_ent = None
        preds = []
        for t in range(sample.size(1)):
            inp_t = sample[:, t].unsqueeze(1)
            logits, mem_x, mem_ent, _ = model(inp_t, mem_x, mem_ent)
            p = torch.softmax(logits.squeeze(0).squeeze(0), dim=-1)
            preds.append(int(p.argmax().item()))
        exp["predictions"] = preds
        exp["ground_truth"] = target
    # plot
    plt.figure()
    plt.plot(range(len(preds)), preds, label="predicted")
    plt.plot(range(len(target)), target, label="ground_truth")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"synthetic_generation_{key}.png"))
    plt.close()
    # store
    experiment_data["learning_rate"][key] = {"synthetic": exp}

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
