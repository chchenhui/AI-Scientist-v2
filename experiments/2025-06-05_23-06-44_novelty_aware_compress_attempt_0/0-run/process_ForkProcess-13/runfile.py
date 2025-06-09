import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparams
vocab_size = 256
max_len = 128
embed_dim = 32
num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3


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
            mem = mem_x.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem, x], dim=1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))

        eps = 1e-10
        # attn_w: [B, H, T, S]; entropy per head, per token
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # [B, H, T]
        ent_tok = ent_h[0].max(dim=0)[0]  # [T]

        if self.detach_mem:
            mem_update = x.detach()[0]
        else:
            mem_update = x[0]

        if mem_x is None:
            mem_x_new = mem_update
            mem_ent_new = ent_tok
        else:
            mem_x_new = torch.cat([mem_x, mem_update], dim=0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], dim=0)

        if mem_x_new.size(0) > self.mem_size:
            total_ent = mem_ent_new.sum().item() + eps
            _, idx = torch.topk(mem_ent_new, self.mem_size)
            kept_ent = mem_ent_new[idx].sum().item()
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
            ratio = kept_ent / total_ent
        else:
            ratio = 1.0

        return out, mem_x_new, mem_ent_new, ratio


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
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


def encode_fn(example):
    txt = example.get("text") or example.get("abstract", "")
    txt = txt[: max_len + 1]
    ids = [ord(c) % vocab_size for c in txt]
    if len(ids) < max_len + 1:
        ids += [0] * (max_len + 1 - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


configs = [
    ("pg19", None),
    ("scientific_papers", "arxiv"),
    ("wikitext", "wikitext-2-raw-v1"),
]

experiment_data = {}
ablation_types = ["baseline", "memory_grad_flow"]

for ablation in ablation_types:
    experiment_data[ablation] = {}
    detach_flag = ablation == "baseline"
    for ds_name, cfg in configs:
        key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"\n=== Ablation: {ablation} | Dataset: {key} ===")

        # load small subsets without streaming
        train_ds = load_dataset(ds_name, cfg, split="train")
        train_subset = train_ds.select(range(200))
        train_enc = [encode_fn(x) for x in train_subset]
        train_inputs = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
        train_targets = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=True
        )

        val_split = "validation" if ds_name != "scientific_papers" else "test"
        val_ds = load_dataset(ds_name, cfg, split=val_split)
        val_subset = val_ds.select(range(100))
        val_enc = [encode_fn(x) for x in val_subset]
        val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
        val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)

        model = ImprovedTransformerXLModel(
            vocab_size, embed_dim, num_heads, mem_size, detach_flag
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        dataset_data = {
            "metrics": {
                "Entropy Retention Ratio": {"train": [], "val": []},
                "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(num_epochs):
            # training
            model.train()
            train_loss, tr_ratios, tr_eme = 0.0, [], []
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
                train_loss += acc_loss.item() / (inputs.size(1) / chunk_size)
            avg_tr_loss = train_loss / len(train_loader)
            avg_tr_ratio = sum(tr_ratios) / len(tr_ratios)
            avg_tr_eme = sum(tr_eme) / len(tr_eme)
            dataset_data["losses"]["train"].append(avg_tr_loss)
            dataset_data["metrics"]["Entropy Retention Ratio"]["train"].append(
                avg_tr_ratio
            )
            dataset_data["metrics"]["Entropy-Weighted Memory Efficiency"][
                "train"
            ].append(avg_tr_eme)

            # validation
            model.eval()
            val_loss, vl_ratios, vl_eme = 0.0, [], []
            if epoch == num_epochs - 1:
                preds_all, gts_all = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if epoch == num_epochs - 1:
                        gt_seq = targets.squeeze(0).tolist()
                        pred_seq = []
                    mem_x = mem_ent = None
                    acc_loss = 0.0
                    for i in range(0, inputs.size(1), chunk_size):
                        ic = inputs[:, i : i + chunk_size]
                        tc = targets[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                        vl_ratios.append(ratio)
                        vl_eme.append(mem_ent.sum().item() / mem_ent.numel())
                        if epoch == num_epochs - 1:
                            preds = logits.argmax(-1).squeeze(0).tolist()
                            pred_seq.extend(preds)
                    val_loss += acc_loss.item() / (inputs.size(1) / chunk_size)
                    if epoch == num_epochs - 1:
                        preds_all.append(pred_seq)
                        gts_all.append(gt_seq)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_ratio = sum(vl_ratios) / len(vl_ratios)
            avg_val_eme = sum(vl_eme) / len(vl_eme)
            dataset_data["losses"]["val"].append(avg_val_loss)
            dataset_data["metrics"]["Entropy Retention Ratio"]["val"].append(
                avg_val_ratio
            )
            dataset_data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"].append(
                avg_val_eme
            )
            print(f"Epoch {epoch}: validation_loss = {avg_val_loss:.4f}")
            if epoch == num_epochs - 1:
                dataset_data["predictions"] = preds_all
                dataset_data["ground_truth"] = gts_all

        experiment_data[ablation][key] = dataset_data

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
