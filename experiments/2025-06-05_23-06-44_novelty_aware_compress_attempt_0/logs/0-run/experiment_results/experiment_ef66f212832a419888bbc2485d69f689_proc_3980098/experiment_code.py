import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import itertools
import numpy as np

# Reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size, max_len, embed_dim = 256, 128, 32
num_heads, mem_size, chunk_size = 2, 50, 32
num_epochs, lr = 2, 1e-3


# Memory‐transformer with switchable retention
class ImprovedMemoryTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mem_size, random_retain=False):
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
        self.random_retain = random_retain

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
            total_ent = mem_ent_new.sum().item() + eps
            if self.random_retain:
                perm = torch.randperm(mem_ent_new.size(0), device=mem_ent_new.device)
                idx = perm[: self.mem_size]
            else:
                _, idx = torch.topk(mem_ent_new, self.mem_size)
            kept_ent = mem_ent_new[idx].sum().item()
            ratio = kept_ent / total_ent
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


class TransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size, random_retain=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = ImprovedMemoryTransformerLayer(
            embed_dim, num_heads, mem_size, random_retain
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


# Datasets and encoding
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


# Experiment storage
ablation_types = ["entropy_based", "random_retention"]
experiment_data = {ab: {} for ab in ablation_types}

for ablation in ablation_types:
    random_retain = ablation == "random_retention"
    for ds_name, cfg in configs:
        dataset_key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"\n=== Ablation: {ablation}, Dataset: {dataset_key} ===")

        # Prepare data
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

        # Model & optimizer
        model = TransformerXLModel(
            vocab_size, embed_dim, num_heads, mem_size, random_retain
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Storage init
        experiment_data[ablation][dataset_key] = {
            "metrics": {
                "Memory Retention Ratio": {"train": [], "val": []},
                "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        # Train & validate
        for epoch in range(num_epochs):
            model.train()
            train_loss, train_ratios, train_eme = 0.0, [], []
            for batch in train_loader:
                inp, tgt = [t.to(device) for t in batch]
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
            ed = experiment_data[ablation][dataset_key]
            ed["losses"]["train"].append(avg_tr_loss)
            ed["metrics"]["Memory Retention Ratio"]["train"].append(avg_tr_ratio)
            ed["metrics"]["Entropy-Weighted Memory Efficiency"]["train"].append(
                avg_tr_eme
            )

            model.eval()
            val_loss, val_ratios, val_eme = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    inp, tgt = [t.to(device) for t in batch]
                    mem_x = mem_ent = None
                    acc_loss = 0.0
                    preds = []
                    for i in range(0, inp.size(1), chunk_size):
                        ic = inp[:, i : i + chunk_size]
                        tc = tgt[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                        val_ratios.append(ratio)
                        val_eme.append(mem_ent.sum().item() / mem_ent.numel())
                        p = logits.argmax(dim=-1)[0].cpu()
                        preds.append(p)
                    val_loss += acc_loss.item() / (inp.size(1) / chunk_size)
                    # record preds & ground truth
                    batch_pred = torch.cat(preds, dim=0).numpy()
                    batch_tgt = tgt[0].cpu().numpy()
                    ed["predictions"].append(batch_pred)
                    ed["ground_truth"].append(batch_tgt)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_ratio = sum(val_ratios) / len(val_ratios)
            avg_val_eme = sum(val_eme) / len(val_eme)
            ed["losses"]["val"].append(avg_val_loss)
            ed["metrics"]["Memory Retention Ratio"]["val"].append(avg_val_ratio)
            ed["metrics"]["Entropy-Weighted Memory Efficiency"]["val"].append(
                avg_val_eme
            )
            print(f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}")

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
