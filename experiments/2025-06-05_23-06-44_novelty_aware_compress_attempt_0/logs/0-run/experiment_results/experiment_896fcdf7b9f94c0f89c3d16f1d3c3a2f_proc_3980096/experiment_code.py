import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import itertools
import numpy as np

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 256
max_len = 128
embed_dim = 32
num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3

# container for results
experiment_data = {"no_layernorm": {}}


# ablated memory layer without any LayerNorm
class MemoryTransformerLayerNoLN(nn.Module):
    def __init__(self, embed_dim, num_heads, mem_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
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
        # skip all LayerNorms
        x2 = x + attn_out
        out = x2 + self.ff(x2)
        # entropy
        eps = 1e-10
        ent_h = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # B, heads, T
        ent_tok = ent_h[0].max(dim=0)[0]  # T
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


# ablated Transformer-XL style model
class AblatedTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = MemoryTransformerLayerNoLN(embed_dim, num_heads, mem_size)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mem_x_new, mem_ent_new, ratio = self.mem_layer(emb, mem_x, mem_ent)
        logits = self.out(out)
        return logits, mem_x_new, mem_ent_new, ratio


# simple char→id encoder
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

for ds_name, cfg in configs:
    key = ds_name if cfg is None else f"{ds_name}_{cfg}"
    print(f"=== Ablation no_layernorm on {key} ===")
    experiment_data["no_layernorm"][key] = {
        "metrics": {
            "Memory Retention Ratio": {"train": [], "val": []},
            "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # prepare data
    train_stream = load_dataset(ds_name, cfg, split="train", streaming=True)
    train_samples = list(itertools.islice(train_stream, 200))
    train_enc = [encode_fn(x) for x in train_samples]
    tr_in = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
    tr_tg = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
    train_loader = DataLoader(TensorDataset(tr_in, tr_tg), batch_size=1, shuffle=True)

    val_split = "validation" if ds_name != "scientific_papers" else "test"
    val_stream = load_dataset(ds_name, cfg, split=val_split, streaming=True)
    val_samples = list(itertools.islice(val_stream, 100))
    val_enc = [encode_fn(x) for x in val_samples]
    vl_in = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
    vl_tg = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
    val_loader = DataLoader(TensorDataset(vl_in, vl_tg), batch_size=1)

    # model, optimizer, loss
    model = AblatedTransformerXLModel(vocab_size, embed_dim, num_heads, mem_size).to(
        device
    )
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # for final-epoch preds
    val_preds, val_gts = [], []

    # training & validation
    for epoch in range(num_epochs):
        # train
        model.train()
        train_loss, train_ratios, train_eme = 0.0, [], []
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            mem_x = mem_ent = None
            opt.zero_grad()
            acc_loss = 0.0
            for i in range(0, inp.size(1), chunk_size):
                ic = inp[:, i : i + chunk_size]
                tc = tgt[:, i : i + chunk_size]
                logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                loss = crit(logits.view(-1, vocab_size), tc.view(-1))
                acc_loss += loss
                train_ratios.append(ratio)
                eme = mem_ent.sum().item() / mem_ent.numel()
                train_eme.append(eme)
            acc_loss.backward()
            opt.step()
            train_loss += acc_loss.item() / (inp.size(1) / chunk_size)
        avg_tr_loss = train_loss / len(train_loader)
        avg_tr_ratio = sum(train_ratios) / len(train_ratios)
        avg_tr_eme = sum(train_eme) / len(train_eme)
        md = experiment_data["no_layernorm"][key]
        md["losses"]["train"].append(avg_tr_loss)
        md["metrics"]["Memory Retention Ratio"]["train"].append(avg_tr_ratio)
        md["metrics"]["Entropy-Weighted Memory Efficiency"]["train"].append(avg_tr_eme)

        # val
        model.eval()
        val_loss, val_ratios, val_eme = 0.0, [], []
        record = epoch == num_epochs - 1
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_ent = None
                acc_loss = 0.0
                if record:
                    pred_tokens, true_tokens = [], []
                for i in range(0, inp.size(1), chunk_size):
                    ic = inp[:, i : i + chunk_size]
                    tc = tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    acc_loss += crit(logits.view(-1, vocab_size), tc.view(-1))
                    val_ratios.append(ratio)
                    eme = mem_ent.sum().item() / mem_ent.numel()
                    val_eme.append(eme)
                    if record:
                        pred_tokens.extend(logits.argmax(-1)[0].tolist())
                        true_tokens.extend(tc[0].tolist())
                val_loss += acc_loss.item() / (inp.size(1) / chunk_size)
                if record:
                    val_preds.append(pred_tokens)
                    val_gts.append(true_tokens)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ratio = sum(val_ratios) / len(val_ratios)
        avg_val_eme = sum(val_eme) / len(val_eme)
        md["losses"]["val"].append(avg_val_loss)
        md["metrics"]["Memory Retention Ratio"]["val"].append(avg_val_ratio)
        md["metrics"]["Entropy-Weighted Memory Efficiency"]["val"].append(avg_val_eme)
        print(f"{key} Epoch {epoch}: val_loss={avg_val_loss:.4f}")
    # store final predictions & ground truth
    experiment_data["no_layernorm"][key]["predictions"] = val_preds
    experiment_data["no_layernorm"][key]["ground_truth"] = val_gts

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
