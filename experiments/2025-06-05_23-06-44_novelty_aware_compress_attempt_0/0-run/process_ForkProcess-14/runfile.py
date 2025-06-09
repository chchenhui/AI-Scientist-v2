import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device setup
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


# memory-aware transformer layer
class ImprovedMemoryTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mem_size, detach_mem=True):
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
        self.detach_mem = detach_mem

    def forward(self, x, mem_x, mem_ent):
        B, T, E = x.size()
        if mem_x is None:
            k = v = x
        else:
            mem = mem_x.unsqueeze(0).expand(B, -1, -1).to(device)
            k = torch.cat([mem, x], dim=1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))

        # entropy per token (max over heads)
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # (B, heads, T)
        ent_tok = ent_h[0].max(dim=0)[0]  # (T,)

        # memory update
        mem_update = x.detach()[0] if self.detach_mem else x[0]
        if mem_x is None:
            mem_x_new, mem_ent_new = mem_update, ent_tok
        else:
            mem_x_new = torch.cat([mem_x, mem_update], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)

        # prune if over budget
        if mem_x_new.size(0) > self.mem_size:
            total_ent = mem_ent_new.sum().item() + eps
            kept_ent = mem_ent_new.topk(self.mem_size)[0].sum().item()
            idx = torch.topk(mem_ent_new, self.mem_size)[1]
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
            ratio = kept_ent / total_ent
        else:
            ratio = 1.0

        return out, mem_x_new, mem_ent_new, ratio


# full model
class ImprovedTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size, detach_mem=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = ImprovedMemoryTransformerLayer(
            embed_dim, num_heads, mem_size, detach_mem
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x, mem_ent, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x, mem_ent, ratio


# simple encoder for text fields
def encode_fn(example):
    txt = example.get("text") or example.get("abstract", "")
    txt = txt[: max_len + 1]
    ids = [ord(c) % vocab_size for c in txt]
    if len(ids) < max_len + 1:
        ids += [0] * (max_len + 1 - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


# datasets to run
configs = [
    ("pg19", None),
    ("scientific_papers", "arxiv"),
    ("wikitext", "wikitext-2-raw-v1"),
]

# preload all splits once without streaming to allow slicing
preloaded = {}
for ds_name, cfg in configs:
    key = ds_name if cfg is None else f"{ds_name}_{cfg}"
    # training split
    train_split = f"train[:200]"
    raw_train = load_dataset(ds_name, cfg, split=train_split)
    train_enc = [encode_fn(x) for x in raw_train]
    train_inputs = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
    train_targets = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=True
    )
    # validation split
    val_base = "validation" if ds_name != "scientific_papers" else "test"
    val_split = f"{val_base}[:100]"
    raw_val = load_dataset(ds_name, cfg, split=val_split)
    val_enc = [encode_fn(x) for x in raw_val]
    val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
    val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)
    preloaded[key] = {"train": train_loader, "val": val_loader}

# main experiment loop
ablation_types = ["baseline", "memory_grad_flow"]
experiment_data = {}

for ablation in ablation_types:
    detach_flag = ablation == "baseline"
    experiment_data[ablation] = {}
    for ds_name, cfg in configs:
        key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"\n=== Ablation: {ablation} | Dataset: {key} ===")

        # fetch preloaded loaders
        train_loader = preloaded[key]["train"]
        val_loader = preloaded[key]["val"]

        # init model and optimizer
        model = ImprovedTransformerXLModel(
            vocab_size, embed_dim, num_heads, mem_size, detach_flag
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # storage
        dataset_data = {
            "metrics": {
                "Memory Retention Ratio": {"train": [], "val": []},
                "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        # train & eval
        for epoch in range(num_epochs):
            # training
            model.train()
            tr_ratios, tr_eme, total_tr_loss = [], [], 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                mem_x = mem_ent = None
                optimizer.zero_grad()
                acc_loss = 0.0
                for i in range(0, inputs.size(1), chunk_size):
                    ic = inputs[:, i : i + chunk_size]
                    tc = targets[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                    acc_loss += loss
                    tr_ratios.append(ratio)
                    tr_eme.append(mem_ent.sum().item() / mem_ent.numel())
                acc_loss.backward()
                optimizer.step()
                total_tr_loss += acc_loss.item() / (inputs.size(1) / chunk_size)
            avg_tr_loss = total_tr_loss / len(train_loader)
            dataset_data["losses"]["train"].append(avg_tr_loss)
            dataset_data["metrics"]["Memory Retention Ratio"]["train"].append(
                sum(tr_ratios) / len(tr_ratios)
            )
            dataset_data["metrics"]["Entropy-Weighted Memory Efficiency"][
                "train"
            ].append(sum(tr_eme) / len(tr_eme))

            # validation
            model.eval()
            vl_ratios, vl_eme, total_val_loss = [], [], 0.0
            preds_all, gts_all = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    mem_x = mem_ent = None
                    acc_loss = 0.0
                    gt_seq = targets.squeeze(0).tolist()
                    pred_seq = []
                    for i in range(0, inputs.size(1), chunk_size):
                        ic = inputs[:, i : i + chunk_size]
                        tc = targets[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                        vl_ratios.append(ratio)
                        vl_eme.append(mem_ent.sum().item() / mem_ent.numel())
                        pred_seq.extend(logits.argmax(-1).squeeze(0).tolist())
                    total_val_loss += acc_loss.item() / (inputs.size(1) / chunk_size)
                    preds_all.append(pred_seq)
                    gts_all.append(gt_seq)
            avg_val_loss = total_val_loss / len(val_loader)
            dataset_data["losses"]["val"].append(avg_val_loss)
            dataset_data["metrics"]["Memory Retention Ratio"]["val"].append(
                sum(vl_ratios) / len(vl_ratios)
            )
            dataset_data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"].append(
                sum(vl_eme) / len(vl_eme)
            )
            print(
                f"{ablation}-{key} Epoch {epoch}: validation_loss = {avg_val_loss:.4f}"
            )

            if epoch == num_epochs - 1:
                dataset_data["predictions"] = preds_all
                dataset_data["ground_truth"] = gts_all

        experiment_data[ablation][key] = dataset_data

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
