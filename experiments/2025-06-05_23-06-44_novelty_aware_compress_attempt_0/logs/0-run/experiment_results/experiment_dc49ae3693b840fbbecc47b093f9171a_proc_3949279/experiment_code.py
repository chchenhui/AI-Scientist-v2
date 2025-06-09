import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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
seq_total = 51
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
        # entropy per query
        aw = attn_w.mean(dim=1)[0]  # (tgt_len, src_len)
        eps = 1e-10
        ent = -(aw * (aw + eps).log()).sum(dim=-1).detach()  # (tgt_len,)
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new, mem_ent_new = x_det, ent
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent], dim=0)
        total_ent = mem_ent_new.sum().item() + eps
        if mem_x_new.size(0) > self.mem_size:
            idx = torch.topk(mem_ent_new, self.mem_size).indices
            kept_ent = mem_ent_new[idx].sum().item()
            ratio = kept_ent / total_ent
            mem_x_new, mem_ent_new = mem_x_new[idx], mem_ent_new[idx]
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


# Hyperparameter tuning over num_heads
num_heads_list = [2, 4, 8]
num_epochs = 3
chunk_size = 10
embed_dim = 64
mem_size = 20
lr = 1e-3

experiment_data = {
    "num_heads_tuning": {
        "synthetic": {
            "num_heads": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for nh in num_heads_list:
    print(f"\n=== Running num_heads = {nh} ===")
    experiment_data["num_heads_tuning"]["synthetic"]["num_heads"].append(nh)

    model = TransformerXLModel(vocab_size, embed_dim, nh, mem_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []

    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss, ratios = 0.0, []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inp, tgt = batch["input"], batch["target"]
            mem_x = mem_ent = None
            optimizer.zero_grad()
            acc_loss = 0.0
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                acc_loss += loss
                ratios.append(ratio)
            acc_loss.backward()
            optimizer.step()
            running_loss += acc_loss.item() / (inp.size(1) / chunk_size)
        avg_loss = running_loss / len(train_loader)
        avg_ratio = sum(ratios) / len(ratios)
        train_losses.append(avg_loss)
        train_metrics.append(avg_ratio)

        # validation
        model.eval()
        running_loss, ratios = 0.0, []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                inp, tgt = batch["input"], batch["target"]
                mem_x = mem_ent = None
                acc_loss = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                    ratios.append(ratio)
                running_loss += acc_loss.item() / (inp.size(1) / chunk_size)
        avg_vloss = running_loss / len(val_loader)
        avg_vratio = sum(ratios) / len(ratios)
        val_losses.append(avg_vloss)
        val_metrics.append(avg_vratio)

        print(
            f"heads={nh} epoch={epoch} train_loss={avg_loss:.4f} val_loss={avg_vloss:.4f}"
        )

    # record per-head results
    sd = experiment_data["num_heads_tuning"]["synthetic"]
    sd["losses"]["train"].append(train_losses)
    sd["losses"]["val"].append(val_losses)
    sd["metrics"]["train"].append(train_metrics)
    sd["metrics"]["val"].append(val_metrics)

    # generate a sample prediction
    model.eval()
    with torch.no_grad():
        sample = val_ds[0]["input"].unsqueeze(0).to(device)
        target = val_ds[0]["target"].tolist()
        preds = []
        mem_x = mem_ent = None
        for t in range(sample.size(1)):
            inp_t = sample[:, t].unsqueeze(1)
            logits, mem_x, mem_ent, _ = model(inp_t, mem_x, mem_ent)
            p = torch.softmax(logits.view(-1), dim=-1)
            preds.append(int(p.argmax().item()))
    sd["predictions"].append(preds)
    sd["ground_truth"].append(target)

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment_data to {working_dir}/experiment_data.npy")
