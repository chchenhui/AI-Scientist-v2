import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# generate synthetic code snippets and execution traces
codes = []
for c in range(1, 11):
    codes.append(f"def f(x): return x+{c}")
    codes.append(f"def f(x): return {c}+x")
input_set = np.random.randint(-10, 10, size=100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in input_set))

# group code snippets by identical trace
trace_to_indices = {}
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = gid

# character‐level encoding
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi[c] for c in s] + [0] * (max_len - len(s)) for s in codes]
).to(device)


# dataset & loaders
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


dataset = CodeDataset(encoded, group_to_indices, index_to_gid)
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)


# encoder variants
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


class MeanPoolEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x):
        emb = self.embed(x)  # [B, L, D]
        mask = (x != 0).unsqueeze(-1).float()  # [B, L, 1]
        summed = (emb * mask).sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        return summed / lengths


# experimental sweep
EPOCH_LIST = [10, 30, 50]
ablations = {"lstm": LSTMEncoder, "mean_pool": MeanPoolEncoder}

experiment_data = {name: {"synthetic": {}} for name in ablations}

for name, Encoder in ablations.items():
    for E in EPOCH_LIST:
        # instantiate model with or without hidden
        if name == "lstm":
            model = Encoder(len(stoi), embed_dim=64, hidden=64).to(device)
        else:
            model = Encoder(len(stoi), embed_dim=64).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0)

        data = {
            "metrics": {
                "train": [],
                "val": [],
                "align_gap_train": [],
                "align_gap_val": [],
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(1, E + 1):
            # training
            model.train()
            tot_train_loss = 0.0
            for a, p, n in train_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                emb_a, emb_p, emb_n = model(a), model(p), model(n)
                loss = loss_fn(emb_a, emb_p, emb_n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_train_loss += loss.item()
            train_loss = tot_train_loss / len(train_loader)
            data["losses"]["train"].append(train_loss)

            # validation & alignment gap
            model.eval()
            tot_val_loss = 0.0
            train_pos, train_neg = [], []
            val_pos, val_neg = [], []
            with torch.no_grad():
                for a, p, n in train_loader:
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    e_a, e_p, e_n = model(a), model(p), model(n)
                    train_pos.append(F.cosine_similarity(e_a, e_p, dim=1))
                    train_neg.append(F.cosine_similarity(e_a, e_n, dim=1))
                for a, p, n in val_loader:
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    e_a, e_p, e_n = model(a), model(p), model(n)
                    tot_val_loss += loss_fn(e_a, e_p, e_n).item()
                    val_pos.append(F.cosine_similarity(e_a, e_p, dim=1))
                    val_neg.append(F.cosine_similarity(e_a, e_n, dim=1))
            val_loss = tot_val_loss / len(val_loader)
            data["losses"]["val"].append(val_loss)

            train_pos = torch.cat(train_pos)
            train_neg = torch.cat(train_neg)
            val_pos = torch.cat(val_pos)
            val_neg = torch.cat(val_neg)
            align_gap_train = train_pos.mean().item() - train_neg.mean().item()
            align_gap_val = val_pos.mean().item() - val_neg.mean().item()
            data["metrics"]["align_gap_train"].append(align_gap_train)
            data["metrics"]["align_gap_val"].append(align_gap_val)

            # retrieval accuracy
            with torch.no_grad():
                emb_all = model(encoded)
                emb_norm = F.normalize(emb_all, dim=1)
                sims = emb_norm @ emb_norm.T

                def compute_acc(idxs):
                    c = 0
                    for i in idxs:
                        s = sims[i].clone()
                        s[i] = -1e9
                        top1 = torch.argmax(s).item()
                        if index_to_gid[top1] == index_to_gid[i]:
                            c += 1
                    return c / len(idxs)

                train_acc = compute_acc(train_indices)
                val_acc = compute_acc(val_indices)
            data["metrics"]["train"].append(train_acc)
            data["metrics"]["val"].append(val_acc)

            print(
                f"Finished {name}, E={E}, Epoch {epoch}: validation_loss = {val_loss:.4f}, val_align_gap = {align_gap_val:.4f}"
            )

        # final predictions on val set
        model.eval()
        with torch.no_grad():
            emb_all = model(encoded)
            emb_norm = F.normalize(emb_all, dim=1)
            sims = emb_norm @ emb_norm.T
            for i in val_indices:
                s = sims[i].clone()
                s[i] = -1e9
                pick = torch.argmax(s).item()
                data["predictions"].append(index_to_gid[pick])
                data["ground_truth"].append(index_to_gid[i])

        experiment_data[name]["synthetic"][E] = data

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
