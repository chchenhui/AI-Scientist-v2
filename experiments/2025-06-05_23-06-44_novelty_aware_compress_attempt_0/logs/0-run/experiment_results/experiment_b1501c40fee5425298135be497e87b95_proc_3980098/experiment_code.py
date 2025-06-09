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
print("Using device:", device)

# hyperparams
vocab_size, max_len = 256, 128
embed_dim, num_heads, mem_size = 32, 2, 50
chunk_size, num_epochs, lr = 32, 2, 1e-3


# memory layer & model
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
            k = torch.cat([mem, x], 1)
            v = k
        attn_out, attn_w = self.attn(
            x, k, v, need_weights=True, average_attn_weights=False
        )
        x2 = self.norm1(x + attn_out)
        out = self.norm2(x2 + self.ff(x2))
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(-1)  # B,heads,T
        ent_tok = ent_h[0].max(0)[0]  # T
        x_det = x.detach()[0]
        if mem_x is None:
            mem_x_new, mem_ent_new = x_det, ent_tok
        else:
            mem_x_new = torch.cat([mem_x, x_det], 0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], 0)
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


# data configs & encoder
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


# experiment container
experiment_data = {}

for ablation in ["baseline", "continuous_memory"]:
    experiment_data[ablation] = {}
    for ds_name, cfg in configs:
        key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"\n--- Ablation: {ablation}, Dataset: {key} ---")
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
        vsplit = "validation" if ds_name != "scientific_papers" else "test"
        val_stream = load_dataset(ds_name, cfg, split=vsplit, streaming=True)
        val_samples = list(itertools.islice(val_stream, 100))
        val_enc = [encode_fn(x) for x in val_samples]
        val_inputs = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
        val_targets = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=1)

        # model, optimizer, loss
        model = ImprovedTransformerXLModel(
            vocab_size, embed_dim, num_heads, mem_size
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # init storage
        experiment_data[ablation][key] = {
            "metrics": {
                "Memory Retention Ratio": {"train": [], "val": []},
                "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        # epochs
        for epoch in range(num_epochs):
            print(f"Abl {ablation}, {key} — Epoch {epoch}")
            # TRAIN
            model.train()
            train_loss, train_ratios, train_eme = 0.0, [], []
            if ablation == "continuous_memory":
                mem_x_train, mem_ent_train = None, None
            for batch in train_loader:
                inp, tgt = batch[0].to(device), batch[1].to(device)
                if ablation == "baseline":
                    mem_x, mem_ent = None, None
                else:
                    mem_x, mem_ent = mem_x_train, mem_ent_train
                optimizer.zero_grad()
                acc_loss = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                    acc_loss += loss
                    train_ratios.append(ratio)
                    eme = mem_ent.sum().item() / mem_ent.numel()
                    train_eme.append(eme)
                acc_loss.backward()
                optimizer.step()
                train_loss += acc_loss.item() / (inp.size(1) / chunk_size)
                if ablation == "continuous_memory":
                    mem_x_train, mem_ent_train = mem_x, mem_ent
            avg_tr_loss = train_loss / len(train_loader)
            avg_tr_ratio = sum(train_ratios) / len(train_ratios)
            avg_tr_eme = sum(train_eme) / len(train_eme)
            experiment_data[ablation][key]["losses"]["train"].append(avg_tr_loss)
            experiment_data[ablation][key]["metrics"]["Memory Retention Ratio"][
                "train"
            ].append(avg_tr_ratio)
            experiment_data[ablation][key]["metrics"][
                "Entropy-Weighted Memory Efficiency"
            ]["train"].append(avg_tr_eme)

            # VAL
            model.eval()
            val_loss, val_ratios, val_eme = 0.0, [], []
            record_preds = epoch == num_epochs - 1
            if record_preds:
                val_preds, val_gts = [], []
            if ablation == "continuous_memory":
                mem_x_val, mem_ent_val = None, None
            with torch.no_grad():
                for batch in val_loader:
                    inp, tgt = batch[0].to(device), batch[1].to(device)
                    if ablation == "baseline":
                        mem_x, mem_ent = None, None
                    else:
                        mem_x, mem_ent = mem_x_val, mem_ent_val
                    acc_loss = 0.0
                    seq_p, seq_g = [], []
                    for i in range(0, inp.size(1), chunk_size):
                        ic = inp[:, i : i + chunk_size]
                        tc = tgt[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                        acc_loss += loss
                        val_ratios.append(ratio)
                        eme = mem_ent.sum().item() / mem_ent.numel()
                        val_eme.append(eme)
                        if record_preds:
                            seq_p.extend(logits.argmax(-1).squeeze(0).cpu().tolist())
                            seq_g.extend(tc.squeeze(0).cpu().tolist())
                    val_loss += acc_loss.item() / (inp.size(1) / chunk_size)
                    if record_preds:
                        val_preds.append(seq_p)
                        val_gts.append(seq_g)
                    if ablation == "continuous_memory":
                        mem_x_val, mem_ent_val = mem_x, mem_ent
            avg_val_loss = val_loss / len(val_loader)
            avg_val_ratio = sum(val_ratios) / len(val_ratios)
            avg_val_eme = sum(val_eme) / len(val_eme)
            experiment_data[ablation][key]["losses"]["val"].append(avg_val_loss)
            experiment_data[ablation][key]["metrics"]["Memory Retention Ratio"][
                "val"
            ].append(avg_val_ratio)
            experiment_data[ablation][key]["metrics"][
                "Entropy-Weighted Memory Efficiency"
            ]["val"].append(avg_val_eme)
            print(f"  val_loss={avg_val_loss:.4f}")

        # save final preds & gts
        experiment_data[ablation][key]["predictions"] = np.array(val_preds)
        experiment_data[ablation][key]["ground_truth"] = np.array(val_gts)

# save all
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
