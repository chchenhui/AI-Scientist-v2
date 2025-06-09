import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import itertools
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparameters
vocab_size = 256
max_len = 128
embed_dim = 32
num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3


class ImprovedMemoryTransformerLayer(nn.Module):
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
            v = torch.cat([mem, x], dim=1)
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # B, heads, T
        ent_tok = ent_h[0].max(dim=0)[0]
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new = x_det
            mem_ent_new = ent_tok
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)
        if mem_x_new.size(0) > self.mem_size:
            total_ent = mem_ent_new.sum().item() + eps
            _, idx = torch.topk(mem_ent_new, self.mem_size)
            kept_ent = mem_ent_new[idx].sum().item()
            ratio = kept_ent / total_ent
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


class KeyOnlyMemoryTransformerLayer(nn.Module):
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
            pad = torch.zeros_like(mem)
            v = torch.cat([pad, x], dim=1)
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
            mem_x_new = x_det
            mem_ent_new = ent_tok
        else:
            mem_x_new = torch.cat([mem_x, x_det], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)
        if mem_x_new.size(0) > self.mem_size:
            total_ent = mem_ent_new.sum().item() + eps
            _, idx = torch.topk(mem_ent_new, self.mem_size)
            kept_ent = mem_ent_new[idx].sum().item()
            ratio = kept_ent / total_ent
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


class ImprovedTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = ImprovedMemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


class KeyOnlyTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = KeyOnlyMemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


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
        ids += [0] * (max_len + 1 - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


experiment_data = {}
ablations = {
    "baseline": ImprovedTransformerXLModel,
    "memory_key_only": KeyOnlyTransformerXLModel,
}
metric_names = ["Entropy Retention Ratio", "Entropy-Weighted Memory Efficiency"]

for ablation_name, ModelClass in ablations.items():
    experiment_data[ablation_name] = {}
    for ds_name, cfg in configs:
        ds_key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"\n=== Ablation: {ablation_name}, Dataset: {ds_key} ===")
        train_stream = load_dataset(ds_name, cfg, split="train", streaming=True)
        train_samples = list(itertools.islice(train_stream, 200))
        train_enc = [encode_fn(x) for x in train_samples]
        train_inputs = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
        train_targets = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=True
        )

        val_split = "validation" if ds_name != "scientific_papers" else "test"
        val_stream = load_dataset(ds_name, cfg, split=val_split, streaming=True)
        val_samples = list(itertools.islice(val_stream, 100))
        val_enc = [encode_fn(x) for x in val_samples]
        val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
        val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)

        model = ModelClass(vocab_size, embed_dim, num_heads, mem_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        data_dict = {
            "metrics": {m: {"train": [], "val": []} for m in metric_names},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_ratios, train_eme = [], []
            for inp, tgt in train_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_ent = None
                optimizer.zero_grad()
                acc_loss = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                    acc_loss += loss
                    train_ratios.append(ratio)
                    train_eme.append(mem_ent.sum().item() / mem_ent.numel())
                acc_loss.backward()
                optimizer.step()
                train_loss += acc_loss.item() / (inp.size(1) / chunk_size)
            avg_tr_loss = train_loss / len(train_loader)
            avg_tr_ratio = sum(train_ratios) / len(train_ratios)
            avg_tr_eme = sum(train_eme) / len(train_eme)
            data_dict["losses"]["train"].append(avg_tr_loss)
            data_dict["metrics"]["Entropy Retention Ratio"]["train"].append(
                avg_tr_ratio
            )
            data_dict["metrics"]["Entropy-Weighted Memory Efficiency"]["train"].append(
                avg_tr_eme
            )

            model.eval()
            val_loss = 0.0
            val_ratios, val_eme = [], []
            preds_epoch, gts_epoch = [], []
            with torch.no_grad():
                for inp, tgt in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    mem_x = mem_ent = None
                    acc_loss = 0.0
                    sample_preds, sample_gts = [], []
                    for i in range(0, inp.size(1), chunk_size):
                        ic = inp[:, i : i + chunk_size]
                        tc = tgt[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                        val_ratios.append(ratio)
                        val_eme.append(mem_ent.sum().item() / mem_ent.numel())
                        sample_preds.extend(logits.argmax(-1).view(-1).cpu().tolist())
                        sample_gts.extend(tc.view(-1).cpu().tolist())
                    val_loss += acc_loss.item() / (inp.size(1) / chunk_size)
                    preds_epoch.append(sample_preds)
                    gts_epoch.append(sample_gts)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_ratio = sum(val_ratios) / len(val_ratios)
            avg_val_eme = sum(val_eme) / len(val_eme)
            data_dict["losses"]["val"].append(avg_val_loss)
            data_dict["metrics"]["Entropy Retention Ratio"]["val"].append(avg_val_ratio)
            data_dict["metrics"]["Entropy-Weighted Memory Efficiency"]["val"].append(
                avg_val_eme
            )
            data_dict["predictions"].append(np.array(preds_epoch))
            data_dict["ground_truth"].append(np.array(gts_epoch))

            print(f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}")

        experiment_data[ablation_name][ds_key] = data_dict

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
