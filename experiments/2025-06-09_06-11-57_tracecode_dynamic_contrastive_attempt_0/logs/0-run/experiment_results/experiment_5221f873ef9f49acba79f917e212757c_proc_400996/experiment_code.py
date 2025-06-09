import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# hyperparameters
EPOCH_LIST = [10, 30, 50]
experiment_data = {"multi_dataset_synthetic_ablation": {}}

# shared input set for tracing
input_set = np.random.randint(0, 20, size=100)

# define dataset names
dataset_names = ["arith", "branch", "loop"]


# triplet dataset
class CodeDataset(Dataset):
    def __init__(self, encoded, group_to_indices, index_to_gid):
        self.encoded = encoded
        self.group_to_indices = group_to_indices
        self.index_to_gid = index_to_gid

    def __len__(self):
        return len(self.index_to_gid)

    def __getitem__(self, idx):
        anchor = self.encoded[idx]
        gid = self.index_to_gid[idx]
        pos = idx
        while pos == idx:
            pos = random.choice(self.group_to_indices[gid])
        neg_gid = random.choice([g for g in self.group_to_indices if g != gid])
        neg = random.choice(self.group_to_indices[neg_gid])
        return anchor, self.encoded[pos], self.encoded[neg]


# LSTM encoder
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# run ablation across datasets
for name in dataset_names:
    # generate synthetic code variants
    codes = []
    if name == "arith":
        for c in range(1, 11):
            codes.append(f"def f(x): return x+{c}")
            codes.append(f"def f(x): return {c}+x")
    elif name == "branch":
        for c in range(1, 11):
            codes.append(
                f"def f(x):\n"
                f"    if x%{c}==0:\n"
                f"        return x//{c}\n"
                f"    else:\n"
                f"        return {c}*x"
            )
            codes.append(f"def f(x): return (x//{c} if x%{c}==0 else {c}*x)")
    elif name == "loop":
        for c in range(1, 11):
            codes.append(
                f"def f(x):\n"
                f"    s=0\n"
                f"    for i in range(x):\n"
                f"        s+=i*{c}\n"
                f"    return s"
            )
            codes.append(
                f"def f(x):\n"
                f"    s=0\n"
                f"    i=0\n"
                f"    while i<x:\n"
                f"        s+=i*{c}\n"
                f"        i+=1\n"
                f"    return s"
            )
    # execute and record traces
    traces = []
    for code in codes:
        env = {}
        exec(code, env)
        f = env["f"]
        traces.append(tuple(f(int(x)) for x in input_set))
    # group by identical trace
    trace_to_indices = {}
    for idx, t in enumerate(traces):
        trace_to_indices.setdefault(t, []).append(idx)
    group_to_indices = {
        gid: idxs for gid, (_, idxs) in enumerate(trace_to_indices.items())
    }
    index_to_gid = [None] * len(codes)
    for gid, idxs in group_to_indices.items():
        for i in idxs:
            index_to_gid[i] = gid
    # encode as character sequences
    vocab = sorted(set("".join(codes)))
    stoi = {c: i + 1 for i, c in enumerate(vocab)}
    stoi["PAD"] = 0
    max_len = max(len(s) for s in codes)
    encoded = torch.LongTensor(
        [[stoi[ch] for ch in s] + [0] * (max_len - len(s)) for s in codes]
    )
    # train/val split on groups
    all_gids = list(group_to_indices.keys())
    random.shuffle(all_gids)
    split = int(0.8 * len(all_gids))
    train_gids, val_gids = all_gids[:split], all_gids[split:]
    train_indices = [i for g in train_gids for i in group_to_indices[g]]
    val_indices = [i for g in val_gids for i in group_to_indices[g]]
    train_loader = DataLoader(
        Subset(CodeDataset(encoded, group_to_indices, index_to_gid), train_indices),
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(CodeDataset(encoded, group_to_indices, index_to_gid), val_indices),
        batch_size=8,
        shuffle=False,
    )
    # run experiments for each epoch count
    per_epoch = {}
    for E in EPOCH_LIST:
        model = CodeEncoder(len(stoi), 64, 64).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        data = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        # train & validate
        for epoch in range(E):
            model.train()
            tot = 0.0
            for a, p, n in train_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                la, lp, ln = model(a), model(p), model(n)
                loss = loss_fn(la, lp, ln)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot += loss.item()
            data["losses"]["train"].append(tot / len(train_loader))
            # val loss
            model.eval()
            totv = 0.0
            with torch.no_grad():
                for a, p, n in val_loader:
                    totv += loss_fn(
                        model(a.to(device)), model(p.to(device)), model(n.to(device))
                    ).item()
            data["losses"]["val"].append(totv / len(val_loader))
            # retrieval acc
            with torch.no_grad():
                emb_all = model(encoded.to(device))
                emb_n = F.normalize(emb_all, dim=1)
                sims = emb_n @ emb_n.T

                def get_acc(idxs):
                    c = 0
                    for i in idxs:
                        v = sims[i].clone()
                        v[i] = -1e9
                        if index_to_gid[torch.argmax(v).item()] == index_to_gid[i]:
                            c += 1
                    return c / len(idxs)

                data["metrics"]["train"].append(get_acc(train_indices))
                data["metrics"]["val"].append(get_acc(val_indices))
        # final top-1 predictions
        model.eval()
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            emb_n = F.normalize(emb_all, dim=1)
            sims = emb_n @ emb_n.T
            for i in val_indices:
                v = sims[i].clone()
                v[i] = -1e9
                data["predictions"].append(index_to_gid[torch.argmax(v).item()])
                data["ground_truth"].append(index_to_gid[i])
        per_epoch[E] = data
        print(
            f"Dataset={name}, EPOCHS={E}, final val_acc={data['metrics']['val'][-1]:.4f}"
        )
    experiment_data["multi_dataset_synthetic_ablation"][name] = {"EPOCHS": per_epoch}

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
