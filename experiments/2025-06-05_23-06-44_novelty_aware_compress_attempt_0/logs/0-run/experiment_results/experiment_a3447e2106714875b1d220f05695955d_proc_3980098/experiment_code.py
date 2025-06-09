import os, random, itertools
import torch, numpy as np
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# hyperparameters
vocab_size = 256
max_len = 128
embed_dim = 32
default_num_heads = 2
mem_size = 50
chunk_size = 32
num_epochs = 2
lr = 1e-3


# model
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
        ent_tok = ent_h[0].max(dim=0)[0]
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
            mem_x_new, mem_ent_new = mem_x_new[idx], mem_ent_new[idx]
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
        out, mx, me, r = self.mem_layer(emb, mem_x, mem_ent)
        return self.out(out), mx, me, r


# ablation helper
def apply_head_ablation(attn_module, num_heads, embed_dim, head_idx):
    head_dim = embed_dim // num_heads
    mask_w = torch.ones(3 * embed_dim, embed_dim)
    mask_b = torch.ones(3 * embed_dim)
    for i in range(3):
        start = i * embed_dim + head_idx * head_dim
        mask_w[start : start + head_dim, :] = 0
        mask_b[start : start + head_dim] = 0
    attn_module.register_buffer("in_proj_weight_mask", mask_w)
    attn_module.register_buffer("in_proj_bias_mask", mask_b)

    def pre_hook(module, _):
        module.in_proj_weight.data.mul_(module.in_proj_weight_mask)
        module.in_proj_bias.data.mul_(module.in_proj_bias_mask)

    attn_module.register_forward_pre_hook(pre_hook)


# encoding and datasets
def encode_fn(example):
    txt = example.get("text") or example.get("abstract", "")
    txt = txt[: max_len + 1]
    ids = [ord(c) % vocab_size for c in txt]
    if len(ids) < max_len + 1:
        ids += [0] * ((max_len + 1) - len(ids))
    return {"input": ids[:-1], "target": ids[1:]}


configs = [
    ("pg19", None),
    ("scientific_papers", "arxiv"),
    ("wikitext", "wikitext-2-raw-v1"),
]

ablation_types = (
    ["baseline"]
    + [f"ablate_head_{i}" for i in range(default_num_heads)]
    + ["single_head"]
)
experiment_data = {}

for ablation in ablation_types:
    experiment_data[ablation] = {}
    num_heads = 1 if ablation == "single_head" else default_num_heads
    for ds_name, cfg in configs:
        key = ds_name if cfg is None else f"{ds_name}_{cfg}"
        print(f"\n=== Ablation {ablation} on {key} ===")

        # prepare data
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

        # init model & ablation
        model = ImprovedTransformerXLModel(vocab_size, embed_dim, num_heads, mem_size)
        if ablation.startswith("ablate_head"):
            head_idx = int(ablation.split("_")[-1])
            apply_head_ablation(model.mem_layer.attn, num_heads, embed_dim, head_idx)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # storage
        experiment_data[ablation][key] = {
            "metrics": {
                "Memory Retention Ratio": {"train": [], "val": []},
                "Entropy-Weighted Memory Efficiency": {"train": [], "val": []},
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        # train & val
        for epoch in range(num_epochs):
            model.train()
            t_loss, t_ratios, t_eme = 0.0, [], []
            for inp, tgt in train_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                mem_x = mem_ent = None
                optimizer.zero_grad()
                acc_loss = 0.0
                for i in range(0, inp.size(1), chunk_size):
                    ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                    logits, mem_x, mem_ent, r = model(ic, mem_x, mem_ent)
                    loss = criterion(logits.view(-1, vocab_size), tc.view(-1))
                    acc_loss += loss
                    t_ratios.append(r)
                    t_eme.append(mem_ent.sum().item() / mem_ent.numel())
                acc_loss.backward()
                optimizer.step()
                t_loss += acc_loss.item() / (inp.size(1) / chunk_size)
            experiment_data[ablation][key]["losses"]["train"].append(
                t_loss / len(train_loader)
            )
            experiment_data[ablation][key]["metrics"]["Memory Retention Ratio"][
                "train"
            ].append(sum(t_ratios) / len(t_ratios))
            experiment_data[ablation][key]["metrics"][
                "Entropy-Weighted Memory Efficiency"
            ]["train"].append(sum(t_eme) / len(t_eme))

            model.eval()
            v_loss, v_ratios, v_eme = 0.0, [], []
            with torch.no_grad():
                for inp, tgt in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    mem_x = mem_ent = None
                    acc_loss = 0.0
                    preds, gts = [], []
                    for i in range(0, inp.size(1), chunk_size):
                        ic, tc = inp[:, i : i + chunk_size], tgt[:, i : i + chunk_size]
                        logits, mem_x, mem_ent, r = model(ic, mem_x, mem_ent)
                        acc_loss += criterion(logits.view(-1, vocab_size), tc.view(-1))
                        v_ratios.append(r)
                        v_eme.append(mem_ent.sum().item() / mem_ent.numel())
                        preds.extend(logits.argmax(dim=-1)[0].cpu().tolist())
                        gts.extend(tc[0].cpu().tolist())
                    v_loss += acc_loss.item() / (inp.size(1) / chunk_size)
                    experiment_data[ablation][key]["predictions"].append(preds)
                    experiment_data[ablation][key]["ground_truth"].append(gts)
            experiment_data[ablation][key]["losses"]["val"].append(
                v_loss / len(val_loader)
            )
            experiment_data[ablation][key]["metrics"]["Memory Retention Ratio"][
                "val"
            ].append(sum(v_ratios) / len(v_ratios))
            experiment_data[ablation][key]["metrics"][
                "Entropy-Weighted Memory Efficiency"
            ]["val"].append(sum(v_eme) / len(v_eme))
            print(f"Epoch {epoch}: val_loss={v_loss/len(val_loader):.4f}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
