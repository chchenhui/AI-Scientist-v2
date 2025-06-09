import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1) Synthetic function variants
group_data = [
    {
        "variants": [
            "def add(a,b): return a+b",
            "def add(x,y): return x+y",
            "def sum_vals(a,b): return b+a",
        ],
        "arity": 2,
    },
    {
        "variants": [
            "def mul(a,b): return a*b",
            "def mul(x,y): return x*y",
            "def multiply(a,b): return b*a",
        ],
        "arity": 2,
    },
    {
        "variants": [
            "def pow2(a): return a**2",
            "def square(x): return x*x",
            "def sq(a): return a*a",
        ],
        "arity": 1,
    },
]

# 2) Generate dynamic traces (not used beyond checking equivalence here)
for grp in group_data:
    # shared random inputs
    if grp["arity"] == 2:
        inputs = [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(16)]
    else:
        inputs = [(random.randint(-10, 10),) for _ in range(16)]
    grp["traces"] = {}
    for code in grp["variants"]:
        local = {}
        exec(code, {}, local)
        fn = next(v for v in local.values() if callable(v))
        outputs = []
        for inp in inputs:
            try:
                outputs.append(fn(*inp))
            except:
                outputs.append(None)
        grp["traces"][code] = tuple(outputs)

# 3) Build triplet dataset
triplets = []
for i, grp in enumerate(group_data):
    other_idxs = [j for j in range(len(group_data)) if j != i]
    for a, b in [
        (x, y)
        for idx, x in enumerate(grp["variants"])
        for y in grp["variants"][idx + 1 :]
    ]:
        neg_grp = group_data[random.choice(other_idxs)]
        neg = random.choice(neg_grp["variants"])
        triplets.append((a, b, neg))
        triplets.append((b, a, neg))


class CodeTripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
        # build char vocab
        chars = set("".join([c for t in triplets for s in t for c in s]))
        self.char2idx = {"<pad>": 0, "<unk>": 1}
        for c in sorted(chars):
            self.char2idx[c] = len(self.char2idx)
        self.pad_idx = 0
        self.unk_idx = 1
        # max seq length
        self.max_len = max(len(s) for t in triplets for s in t)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return self.encode(a), self.encode(p), self.encode(n)

    def encode(self, s):
        arr = [self.char2idx.get(c, self.unk_idx) for c in s]
        if len(arr) < self.max_len:
            arr = arr + [self.pad_idx] * (self.max_len - len(arr))
        return torch.LongTensor(arr)


dataset = CodeTripletDataset(triplets)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# 4) Define encoder model
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, proj_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, proj_dim)

    def forward(self, x):
        emb = self.embed(x)  # (B, L, E)
        mask = (x != 0).unsqueeze(-1).float()  # (B, L, 1)
        emb = emb * mask
        summed = emb.sum(1)  # (B, E)
        lengths = mask.sum(1).clamp(min=1)  # (B,1)
        avg = summed / lengths
        out = self.fc(avg)
        return F.normalize(out, dim=1)


model = CodeEncoder(len(dataset.char2idx)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MarginRankingLoss(margin=1.0)

# 5) Experiment tracking
experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# 6) Training & evaluation
epochs = 10
for epoch in range(epochs):
    model.train()
    t_loss = 0.0
    t_corr = 0
    t_tot = 0
    for a, p, n in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)
        ea, ep, en = model(a), model(p), model(n)
        sim_pos = (ea * ep).sum(1)
        sim_neg = (ea * en).sum(1)
        tgt = torch.ones_like(sim_pos, device=device)
        loss = criterion(sim_pos, sim_neg, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = a.size(0)
        t_loss += loss.item() * bs
        t_corr += (sim_pos > sim_neg).sum().item()
        t_tot += bs
    train_loss = t_loss / t_tot
    train_acc = t_corr / t_tot

    model.eval()
    v_loss = 0.0
    v_corr = 0
    v_tot = 0
    with torch.no_grad():
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ea, ep, en = model(a), model(p), model(n)
            sim_pos = (ea * ep).sum(1)
            sim_neg = (ea * en).sum(1)
            tgt = torch.ones_like(sim_pos, device=device)
            loss = criterion(sim_pos, sim_neg, tgt)
            bs = a.size(0)
            v_loss += loss.item() * bs
            v_corr += (sim_pos > sim_neg).sum().item()
            v_tot += bs
    val_loss = v_loss / v_tot
    val_acc = v_corr / v_tot

    print(
        f"Epoch {epoch+1}: validation_loss = {val_loss:.4f}, Top-1 retrieval accuracy = {val_acc:.4f}"
    )
    experiment_data["synthetic"]["metrics"]["train"].append(
        {"epoch": epoch + 1, "retrieval_accuracy": train_acc}
    )
    experiment_data["synthetic"]["metrics"]["val"].append(
        {"epoch": epoch + 1, "retrieval_accuracy": val_acc}
    )
    experiment_data["synthetic"]["losses"]["train"].append(train_loss)
    experiment_data["synthetic"]["losses"]["val"].append(val_loss)

# 7) Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
