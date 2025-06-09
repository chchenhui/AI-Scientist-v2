import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import itertools
import numpy as np

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparams
vocab_size = 256
max_len = 128
embed_dim = 32
num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3


# Original entropy‐based memory layer
class OriginalMemoryTransformerLayer(nn.Module):
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
        B, T, E = x.size()
        if mem_x is None:
            k = v = x
        else:
            mem = mem_x.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem, x], dim=1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # B,heads,T
        ent_tok = ent_h[0].max(dim=0)[0]  # T
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new = x_det.clone()
            mem_ent_new = ent_tok.clone()
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)
        if mem_x_new.size(0) > self.mem_size:
            total = mem_ent_new.sum().item() + eps
            _, idx = torch.topk(mem_ent_new, self.mem_size)
            kept = mem_ent_new[idx].sum().item()
            ratio = kept / total
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


# Recency‐only memory layer
class RecencyMemoryTransformerLayer(nn.Module):
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
        B, T, E = x.size()
        if mem_x is None:
            k = v = x
        else:
            mem = mem_x.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem, x], dim=1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)
        ent_tok = ent_h[0].max(dim=0)[0]
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new = x_det.clone()
            mem_ent_new = ent_tok.clone()
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)
        if mem_x_new.size(0) > self.mem_size:
            L = mem_x_new.size(0)
            mem_x_new = mem_x_new[-self.mem_size :]
            mem_ent_new = mem_ent_new[-self.mem_size :]
            ratio = self.mem_size / L
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


# Models
class OriginalTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = OriginalMemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mx, me, r = self.mem_layer(emb, mem_x, mem_ent)
        return self.out(out), mx, me, r


class RecencyTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = RecencyMemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mx, me, r = self.mem_layer(emb, mem_x, mem_ent)
        return self.out(out), mx, me, r


# datasets/configs
configs = [
    ("pg19", None),
    ("scientific_papers", "arxiv"),
    ("wikitext", "wikitext-2-raw-v1"),
]


def encode_fn(example):
    txt = example.get("text") or example.get("abstract", "")
    txt = txt[: max_len + 1]
    ids = [ord(c) % vocab_size for c in txt]
    if len(ids) < max_len + 1:
        ids += [0] * ((max_len + 1) - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


# run ablations
experiment_data = {"original": {}, "recency": {}}
mapping = {"original": OriginalTransformerXLModel, "recency": RecencyTransformerXLModel}
for ab_type, ModelClass in mapping.items():
    for ds_name, cfg in configs:
        ds_key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"Running {ab_type} on {ds_key}")
        # load train
        train_stream = load_dataset(ds_name, cfg, split="train", streaming=True)
        train_samples = list(itertools.islice(train_stream, 200))
        train_enc = [encode_fn(x) for x in train_samples]
        train_inputs = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
        train_targets = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=True
        )
        # load val
        vs = "validation" if ds_name != "scientific_papers" else "test"
        val_stream = load_dataset(ds_name, cfg, split=vs, streaming=True)
        val_samples = list(itertools.islice(val_stream, 100))
        val_enc = [encode_fn(x) for x in val_samples]
        val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
        val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)
        # init model, stats
        model = ModelClass(vocab_size, embed_dim, num_heads, mem_size).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        ed = {
            "metrics": {
                "Memory Retention Ratio": {"train": [], "val": []},
                "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        experiment_data[ab_type][ds_key] = ed
        # train/val loop
        for epoch in range(num_epochs):
            # train
            model.train()
            tr_loss, tr_ratios, tr_eme = 0.0, [], []
            for inp, tgt in train_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_ent = None
                opt.zero_grad()
                batch_loss = 0
                for i in range(0, inp.size(1), chunk_size):
                    ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    loss = crit(logits.view(-1, vocab_size), tc.view(-1))
                    batch_loss += loss
                    tr_ratios.append(ratio)
                    tr_eme.append(mem_ent.sum().item() / mem_ent.numel())
                batch_loss.backward()
                opt.step()
                tr_loss += batch_loss.item() / (inp.size(1) / chunk_size)
            avg_tr_loss = tr_loss / len(train_loader)
            avg_tr_ratio = sum(tr_ratios) / len(tr_ratios)
            avg_tr_eme = sum(tr_eme) / len(tr_eme)
            ed["losses"]["train"].append(avg_tr_loss)
            ed["metrics"]["Memory Retention Ratio"]["train"].append(avg_tr_ratio)
            ed["metrics"]["Entropy-Weighted Memory Efficiency"]["train"].append(
                avg_tr_eme
            )
            # val
            model.eval()
            val_loss, val_ratios, val_eme = 0.0, [], []
            with torch.no_grad():
                for bi, (inp, tgt) in enumerate(val_loader):
                    inp, tgt = inp.to(device), tgt.to(device)
                    mem_x = mem_ent = None
                    sample_loss = 0
                    sample_pred, sample_gt = [], []
                    for i in range(0, inp.size(1), chunk_size):
                        ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        sample_loss += crit(logits.view(-1, vocab_size), tc.view(-1))
                        val_ratios.append(ratio)
                        val_eme.append(mem_ent.sum().item() / mem_ent.numel())
                        preds = logits.argmax(-1).squeeze(0).cpu().tolist()
                        sample_pred += preds
                        sample_gt += tc.squeeze(0).cpu().tolist()
                    val_loss += sample_loss.item() / (inp.size(1) / chunk_size)
                    if epoch == num_epochs - 1:
                        ed["predictions"].append(sample_pred)
                        ed["ground_truth"].append(sample_gt)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_ratio = sum(val_ratios) / len(val_ratios)
            avg_val_eme = sum(val_eme) / len(val_eme)
            ed["losses"]["val"].append(avg_val_loss)
            ed["metrics"]["Memory Retention Ratio"]["val"].append(avg_val_ratio)
            ed["metrics"]["Entropy-Weighted Memory Efficiency"]["val"].append(
                avg_val_eme
            )
            print(f"{ab_type} {ds_key} Epoch {epoch} val_loss={avg_val_loss:.4f}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
