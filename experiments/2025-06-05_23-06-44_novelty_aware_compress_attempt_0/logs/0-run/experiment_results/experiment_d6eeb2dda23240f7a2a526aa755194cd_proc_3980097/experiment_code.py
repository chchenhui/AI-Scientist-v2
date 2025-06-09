import os, itertools, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
vocab_size, max_len, embed_dim = 256, 128, 32
num_heads, mem_size, chunk_size = 2, 50, 32
num_epochs, lr = 2, 1e-3


# memory layer with switchable retention
class ImprovedMemoryTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mem_size, retention_mode):
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
        self.mode = retention_mode

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
        x_det = x.detach()[0]  # T,E

        if mem_x is None:
            mem_x_new = x_det
            mem_ent_new = ent_tok
        else:
            mem_x_new = torch.cat([mem_x, x_det], 0)
            mem_ent_new = torch.cat([mem_ent, ent_tok], 0)

        if mem_x_new.size(0) > self.mem_size:
            if self.mode == "entropy":
                scores = mem_ent_new
            else:  # norm
                scores = mem_x_new.norm(p=2, dim=1)
            total = scores.sum().item() + eps
            _, idx = torch.topk(scores, self.mem_size)
            kept = scores[idx].sum().item()
            ratio = kept / total
            mem_x_new = mem_x_new[idx]
            mem_ent_new = mem_ent_new[idx]
        else:
            ratio = 1.0
        return out, mem_x_new, mem_ent_new, ratio


# model wrapper
class ImprovedTransformerXLModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, mem_size, mode):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mem_layer = ImprovedMemoryTransformerLayer(
            embed_dim, num_heads, mem_size, mode
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mem_x, mem_ent):
        emb = self.embed(x)
        out, mx, me, r = self.mem_layer(emb, mem_x, mem_ent)
        return self.out(out), mx, me, r


# datasets
configs = [
    ("pg19", None),
    ("scientific_papers", "arxiv"),
    ("wikitext", "wikitext-2-raw-v1"),
]


def encode_fn(ex):
    txt = ex.get("text") or ex.get("abstract", "")
    txt = txt[: max_len + 1]
    ids = [ord(c) % vocab_size for c in txt]
    if len(ids) < max_len + 1:
        ids += [0] * ((max_len + 1) - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


# initialize experiment_data
ablation_types = ["entropy", "norm"]
experiment_data = {a: {} for a in ablation_types}

# run experiments
for mode in ablation_types:
    for ds_name, cfg in configs:
        key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        # load & encode
        train_stream = load_dataset(ds_name, cfg, split="train", streaming=True)
        train_samples = list(itertools.islice(train_stream, 200))
        train_enc = [encode_fn(x) for x in train_samples]
        ti = torch.tensor([d["input"] for d in train_enc], dtype=torch.long)
        to = torch.tensor([d["target"] for d in train_enc], dtype=torch.long)
        train_loader = DataLoader(TensorDataset(ti, to), batch_size=1, shuffle=True)
        val_split = "validation" if ds_name != "scientific_papers" else "test"
        val_stream = load_dataset(ds_name, cfg, split=val_split, streaming=True)
        val_samples = list(itertools.islice(val_stream, 100))
        val_enc = [encode_fn(x) for x in val_samples]
        vi = torch.tensor([d["input"] for d in val_enc], dtype=torch.long)
        vo = torch.tensor([d["target"] for d in val_enc], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(vi, vo), batch_size=1)

        # storage
        experiment_data[mode][key] = {
            "metrics": {
                "Memory Retention Ratio": {"train": [], "val": []},
                "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        # model & opt
        model = ImprovedTransformerXLModel(
            vocab_size, embed_dim, num_heads, mem_size, mode
        ).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        # train & validate
        for ep in range(num_epochs):
            # train
            model.train()
            tr_loss, tr_ratios, tr_eme = 0, [], []
            for inp, tgt in train_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_ent = None
                opt.zero_grad()
                acc = 0
                for i in range(0, inp.size(1), chunk_size):
                    ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                    loss = crit(logits.view(-1, vocab_size), tc.view(-1))
                    acc += loss
                    tr_ratios.append(ratio)
                    tr_eme.append(mem_ent.sum().item() / mem_ent.numel())
                acc.backward()
                opt.step()
                tr_loss += acc.item() / (inp.size(1) / chunk_size)
            ave_tr_loss = tr_loss / len(train_loader)
            experiment_data[mode][key]["losses"]["train"].append(ave_tr_loss)
            experiment_data[mode][key]["metrics"]["Memory Retention Ratio"][
                "train"
            ].append(sum(tr_ratios) / len(tr_ratios))
            experiment_data[mode][key]["metrics"]["Entropy-Weighted Memory Efficiency"][
                "train"
            ].append(sum(tr_eme) / len(tr_eme))

            # val
            model.eval()
            val_loss, val_ratios, val_eme = 0, [], []
            with torch.no_grad():
                for inp, tgt in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    mem_x = mem_ent = None
                    acc = 0
                    for i in range(0, inp.size(1), chunk_size):
                        ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, ratio = model(ic, mem_x, mem_ent)
                        acc += crit(logits.view(-1, vocab_size), tc.view(-1))
                        val_ratios.append(ratio)
                        val_eme.append(mem_ent.sum().item() / mem_ent.numel())
                        # record preds & gt
                        preds = logits.argmax(-1).cpu().flatten().tolist()
                        gts = tc.cpu().flatten().tolist()
                        experiment_data[mode][key]["predictions"].extend(preds)
                        experiment_data[mode][key]["ground_truth"].extend(gts)
                    val_loss += acc.item() / (inp.size(1) / chunk_size)
            ave_val_loss = val_loss / len(val_loader)
            experiment_data[mode][key]["losses"]["val"].append(ave_val_loss)
            experiment_data[mode][key]["metrics"]["Memory Retention Ratio"][
                "val"
            ].append(sum(val_ratios) / len(val_ratios))
            experiment_data[mode][key]["metrics"]["Entropy-Weighted Memory Efficiency"][
                "val"
            ].append(sum(val_eme) / len(val_eme))
            print(f"[{mode}][{key}] Epoch {ep} val_loss={ave_val_loss:.4f}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
