import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# prepare working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# synthetic dataset
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


# model components
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
        # x: (B,T,E)
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
        # entropy over attention
        aw = attn_w.mean(dim=1)[0]  # (T_src,T_tgt)
        eps = 1e-10
        ent = -(aw * (aw + eps).log()).sum(dim=-1).detach()
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
            kept = mem_ent_new[idx].sum().item()
            ratio = kept / total_ent
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
        out, mx, me, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mx, me, ratio


# hyperparameter grid
embed_dims = [32, 64, 128, 256]
num_heads = 2
mem_size = 20
num_epochs = 3
chunk_size = 10
lr = 1e-3

# collect data
experiment_data = {
    "embed_dim_sweep": {
        "synthetic": {
            "embed_dims": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for ed in embed_dims:
    print(f"=== Sweeping embed_dim={ed} ===")
    model = TransformerXLModel(vocab_size, ed, num_heads, mem_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ratios_by_epoch = []
    val_ratios_by_epoch = []
    train_losses_by_epoch = []
    val_losses_by_epoch = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        all_train_ratios = []
        for batch in train_loader:
            inp, tgt = batch["input"].to(device), batch["target"].to(device)
            mem_x = mem_ent = None
            optimizer.zero_grad()
            loss_accum = 0.0
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                loss_accum = loss_accum + loss
                all_train_ratios.append(ratio)
            loss_accum.backward()
            optimizer.step()
            total_train_loss += loss_accum.item() / (inp.size(1) / chunk_size)
        avg_tr_loss = total_train_loss / len(train_loader)
        avg_tr_ratio = sum(all_train_ratios) / len(all_train_ratios)

        model.eval()
        total_val_loss = 0.0
        all_val_ratios = []
        with torch.no_grad():
            for batch in val_loader:
                inp, tgt = batch["input"].to(device), batch["target"].to(device)
                mem_x = mem_ent = None
                loss_acc = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    loss_acc += criterion(logits.view(-1, vocab_size), tc.view(-1))
                    all_val_ratios.append(ratio)
                total_val_loss += loss_acc.item() / (inp.size(1) / chunk_size)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_ratio = sum(all_val_ratios) / len(all_val_ratios)

        print(
            f"dim={ed} epoch={epoch} tr_loss={avg_tr_loss:.4f} val_loss={avg_val_loss:.4f} val_ratio={avg_val_ratio:.4f}"
        )
        train_losses_by_epoch.append(avg_tr_loss)
        train_ratios_by_epoch.append(avg_tr_ratio)
        val_losses_by_epoch.append(avg_val_loss)
        val_ratios_by_epoch.append(avg_val_ratio)

    # generation on first val sample
    model.eval()
    with torch.no_grad():
        sample = val_ds[0]["input"].unsqueeze(0).to(device)
        target = val_ds[0]["target"].tolist()
        mem_x = mem_ent = None
        preds = []
        for t in range(sample.size(1)):
            inp = sample[:, t].unsqueeze(1)
            logits, mem_x, mem_ent, _ = model(inp, mem_x, mem_ent)
            p = torch.softmax(logits.view(-1), dim=-1)
            preds.append(int(p.argmax().item()))

    d = experiment_data["embed_dim_sweep"]["synthetic"]
    d["embed_dims"].append(ed)
    d["metrics"]["train"].append(train_ratios_by_epoch)
    d["metrics"]["val"].append(val_ratios_by_epoch)
    d["losses"]["train"].append(train_losses_by_epoch)
    d["losses"]["val"].append(val_losses_by_epoch)
    d["predictions"].append(preds)
    d["ground_truth"].append(target)

# convert lists to numpy arrays
d = experiment_data["embed_dim_sweep"]["synthetic"]
d["embed_dims"] = np.array(d["embed_dims"])
d["metrics"]["train"] = np.array(d["metrics"]["train"])
d["metrics"]["val"] = np.array(d["metrics"]["val"])
d["losses"]["train"] = np.array(d["losses"]["train"])
d["losses"]["val"] = np.array(d["losses"]["val"])
d["predictions"] = np.array(d["predictions"])
d["ground_truth"] = np.array(d["ground_truth"])

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
