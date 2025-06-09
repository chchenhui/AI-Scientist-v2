import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import itertools
import numpy as np

# prepare working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
vocab_size = 256
max_len = 128
embed_dim = 32
num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3


# no‐residual memory transformer layer
class ImprovedMemoryTransformerLayerNoRes(nn.Module):
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
        # remove residuals
        x2 = self.norm1(attn_out)
        ff_out = self.ff(x2)
        out = self.norm2(ff_out)

        # compute entropies
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)
        ent_tok = ent_h[0].max(dim=0)[0]
        x_det = x.detach()[0]

        # update memory
        if mem_x is None:
            mem_x_new, mem_ent_new = x_det, ent_tok
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


# model with no‐residual memory layer
class ImprovedTransformerXLModelNoRes(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = ImprovedMemoryTransformerLayerNoRes(
            embed_dim, num_heads, mem_size
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


# simple encoder: char→id
def encode_fn(example):
    txt = example.get("text") or example.get("abstract", "")
    txt = txt[: max_len + 1]
    ids = [ord(c) % vocab_size for c in txt]
    if len(ids) < max_len + 1:
        ids += [0] * (max_len + 1 - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


# datasets
configs = [
    ("pg19", None),
    ("scientific_papers", "arxiv"),
    ("wikitext", "wikitext-2-raw-v1"),
]

# initialize storage
experiment_data = {"residual_connection_ablation": {}}
criterion = nn.CrossEntropyLoss()

for ds_name, cfg in configs:
    key = ds_name if cfg is None else f"{ds_name}_{cfg}"
    experiment_data["residual_connection_ablation"][key] = {
        "metrics": {
            "Memory Retention Ratio": {"train": [], "val": []},
            "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # prepare train
    train_stream = load_dataset(ds_name, cfg, split="train", streaming=True)
    train_samples = list(itertools.islice(train_stream, 200))
    train_enc = [encode_fn(x) for x in train_samples]
    train_inputs = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
    train_targets = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=True
    )

    # prepare val
    val_split = "validation" if ds_name != "scientific_papers" else "test"
    val_stream = load_dataset(ds_name, cfg, split=val_split, streaming=True)
    val_samples = list(itertools.islice(val_stream, 100))
    val_enc = [encode_fn(x) for x in val_samples]
    val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
    val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)

    model = ImprovedTransformerXLModelNoRes(
        vocab_size, embed_dim, num_heads, mem_size
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss, tr_ratios, tr_eme = 0.0, [], []
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            mem_x = mem_ent = None
            optimizer.zero_grad()
            acc_loss = 0.0
            for i in range(0, inp.size(1), chunk_size):
                ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                acc_loss += loss
                tr_ratios.append(ratio)
                tr_eme.append(mem_ent.sum().item() / mem_ent.numel())
            acc_loss.backward()
            optimizer.step()
            tr_loss += acc_loss.item() / (inp.size(1) / chunk_size)
        avg_tr_loss = tr_loss / len(train_loader)
        avg_tr_ratio = sum(tr_ratios) / len(tr_ratios)
        avg_tr_eme = sum(tr_eme) / len(tr_eme)
        ed = experiment_data["residual_connection_ablation"][key]
        ed["losses"]["train"].append(avg_tr_loss)
        ed["metrics"]["Memory Retention Ratio"]["train"].append(avg_tr_ratio)
        ed["metrics"]["Entropy-Weighted Memory Efficiency"]["train"].append(avg_tr_eme)

        # validation + predictions
        model.eval()
        val_loss, val_ratios, val_eme = 0.0, [], []
        preds, gts = [], []
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_ent = None
                acc_loss = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                    val_ratios.append(ratio)
                    val_eme.append(mem_ent.sum().item() / mem_ent.numel())
                    preds.append(logits.argmax(dim=-1).cpu().numpy().flatten())
                    gts.append(tc.cpu().numpy().flatten())
                val_loss += acc_loss.item() / (inp.size(1) / chunk_size)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ratio = sum(val_ratios) / len(val_ratios)
        avg_val_eme = sum(val_eme) / len(val_eme)
        ed["losses"]["val"].append(avg_val_loss)
        ed["metrics"]["Memory Retention Ratio"]["val"].append(avg_val_ratio)
        ed["metrics"]["Entropy-Weighted Memory Efficiency"]["val"].append(avg_val_eme)
        ed["predictions"] = np.concatenate(preds)
        ed["ground_truth"] = np.concatenate(gts)
        print(f"{key} epoch {epoch}: val_loss={avg_val_loss:.4f}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
