import os
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
vocab_size, max_len, embed_dim = 256, 128, 32
num_heads, mem_size, chunk_size = 2, 50, 32
num_epochs, lr = 2, 1e-3


class UsageMemoryTransformerLayer(nn.Module):
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

    def forward(self, x, mem_x, mem_usage):
        B, T, E = x.size()
        if mem_x is None:
            # No prior memory
            k = v = x
            prev_slots = 0
        else:
            prev_slots = mem_x.size(0)
            mem = mem_x.unsqueeze(0).expand(B, -1, -1)
            k = torch.cat([mem, x], dim=1)
            v = k

        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        x_det = x.detach()[0]  # (T, E)

        if mem_x is None:
            mem_x_new = x_det
            mem_usage_new = torch.zeros(T, device=x.device)
            ratio = 1.0
            mem_concat_size = T
        else:
            # accumulate usage from attention on old slots
            w = attn_w[0]  # (heads, T, prev_slots+T)
            w_mem = w[:, :, :prev_slots]
            usage_delta = w_mem.sum(dim=(0, 1))  # (prev_slots,)
            new_usage_existing = mem_usage + usage_delta

            # concatenate old + new slots
            mem_x_concat = torch.cat([mem_x, x_det], dim=0)
            usage_concat = torch.cat(
                [new_usage_existing, torch.zeros(T, device=x.device)], dim=0
            )
            mem_concat_size = mem_x_concat.size(0)

            # prune if over capacity
            if mem_concat_size > self.mem_size:
                _, idx = torch.topk(usage_concat, self.mem_size)
                mem_x_new = mem_x_concat[idx]
                mem_usage_new = usage_concat[idx]
            else:
                mem_x_new = mem_x_concat
                mem_usage_new = usage_concat

            # compute slot‐retention ratio
            ratio = float(mem_x_new.size(0)) / float(mem_concat_size)

        return out, mem_x_new, mem_usage_new, ratio


class UsageTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = UsageMemoryTransformerLayer(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_usage):
        emb = self.embed(x).to(device)
        out, mem_x_new, mem_usage_new, ratio = self.mem_layer(emb, mem_x, mem_usage)
        logits = self.out(out)
        return logits, mem_x_new, mem_usage_new, ratio


# storage for metrics, losses, preds, gts
experiment_data = {"usage_based": {}}
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


for ds_name, cfg in configs:
    key = ds_name if cfg is None else f"{ds_name}_{cfg}"
    print(f"\n=== Dataset: {key} ===")

    # prepare training data
    train_stream = load_dataset(ds_name, cfg, split="train", streaming=True)
    train_samples = list(itertools.islice(train_stream, 200))
    train_enc = [encode_fn(x) for x in train_samples]
    train_inputs = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
    train_targets = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=1, shuffle=True
    )

    # prepare validation data
    val_split = "validation" if ds_name != "scientific_papers" else "test"
    val_stream = load_dataset(ds_name, cfg, split=val_split, streaming=True)
    val_samples = list(itertools.islice(val_stream, 100))
    val_enc = [encode_fn(x) for x in val_samples]
    val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
    val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)

    # init storage
    experiment_data["usage_based"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # model, optimizer, loss
    model = UsageTransformerXLModel(vocab_size, embed_dim, num_heads, mem_size).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training & validation loops
    for epoch in range(num_epochs):
        # training
        model.train()
        train_loss, train_ratios = 0.0, []
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            mem_x = mem_usage = None
            acc_loss = 0.0
            chunks = inp.size(1) // chunk_size
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mem_x, mem_usage, ratio = model(ic, mem_x, mem_usage)
                loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                acc_loss += loss
                train_ratios.append(ratio)
            optimizer.zero_grad()
            acc_loss.backward()
            optimizer.step()
            train_loss += acc_loss.item() / chunks
        avg_tr_loss = train_loss / len(train_loader)
        avg_tr_ratio = sum(train_ratios) / len(train_ratios)
        experiment_data["usage_based"][key]["losses"]["train"].append(avg_tr_loss)
        experiment_data["usage_based"][key]["metrics"]["train"].append(avg_tr_ratio)

        # validation
        model.eval()
        val_loss, val_ratios = 0.0, []
        if epoch == num_epochs - 1:
            experiment_data["usage_based"][key]["predictions"] = []
            experiment_data["usage_based"][key]["ground_truth"] = []
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_usage = None
                acc_loss = 0.0
                chunks = inp.size(1) // chunk_size
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_usage, ratio = model(ic, mem_x, mem_usage)
                    loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                    acc_loss += loss
                    val_ratios.append(ratio)
                    if epoch == num_epochs - 1:
                        preds = logits.argmax(-1)[0].cpu().tolist()
                        gts = tc[0].cpu().tolist()
                        experiment_data["usage_based"][key]["predictions"].extend(preds)
                        experiment_data["usage_based"][key]["ground_truth"].extend(gts)
                val_loss += acc_loss.item() / chunks
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ratio = sum(val_ratios) / len(val_ratios)
        experiment_data["usage_based"][key]["losses"]["val"].append(avg_val_loss)
        experiment_data["usage_based"][key]["metrics"]["val"].append(avg_val_ratio)
        print(
            f"{key} Epoch {epoch} → validation_loss = {avg_val_loss:.4f}, retention = {avg_val_ratio:.4f}"
        )

# save all metrics and results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
