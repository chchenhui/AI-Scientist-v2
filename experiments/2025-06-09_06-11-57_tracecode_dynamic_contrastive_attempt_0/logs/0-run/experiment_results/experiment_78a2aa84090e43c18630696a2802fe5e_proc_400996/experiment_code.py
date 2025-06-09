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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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

# character-level encoding
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = []
for s in codes:
    seq = [stoi[c] for c in s] + [0] * (max_len - len(s))
    encoded.append(seq)
encoded = torch.LongTensor(encoded).to(device)


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
        return {
            "anchor": anchor,
            "pos": self.encoded[pos],
            "neg": self.encoded[neg],
            "idx": idx,
        }


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
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, x):
        emb = self.embed(x)
        mask = (x != 0).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths


# experimental sweep
EPOCH_LIST = [10, 30, 50]
ablations = {
    "lstm": (LSTMEncoder, {"embed_dim": 64, "hidden": 64}),
    "mean_pool": (MeanPoolEncoder, {"embed_dim": 64}),
}
experiment_data = {name: {"synthetic": {}} for name in ablations}

for name, (Encoder, params) in ablations.items():
    for E in EPOCH_LIST:
        model = Encoder(len(stoi), **params).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        data = {
            "metrics": {"train_acc": [], "val_acc": [], "gap_train": [], "gap_val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(1, E + 1):
            # training
            model.train()
            tot_tr, sum_pos, sum_neg, cnt = 0.0, 0.0, 0.0, 0
            for batch in train_loader:
                a = batch["anchor"].to(device)
                p = batch["pos"].to(device)
                n = batch["neg"].to(device)
                emb_a, emb_p, emb_n = model(a), model(p), model(n)
                loss = loss_fn(emb_a, emb_p, emb_n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_tr += loss.item()
                pos_cos = F.cosine_similarity(emb_a, emb_p, dim=1)
                neg_cos = F.cosine_similarity(emb_a, emb_n, dim=1)
                sum_pos += pos_cos.sum().item()
                sum_neg += neg_cos.sum().item()
                cnt += a.size(0)
            train_loss = tot_tr / len(train_loader)
            gap_train = (sum_pos / cnt) - (sum_neg / cnt)
            data["losses"]["train"].append(train_loss)

            # validation
            model.eval()
            tot_val, sum_pos_v, sum_neg_v, cnt_v = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    a = batch["anchor"].to(device)
                    p = batch["pos"].to(device)
                    n = batch["neg"].to(device)
                    emb_a, emb_p, emb_n = model(a), model(p), model(n)
                    tot_val += loss_fn(emb_a, emb_p, emb_n).item()
                    sum_pos_v += F.cosine_similarity(emb_a, emb_p, dim=1).sum().item()
                    sum_neg_v += F.cosine_similarity(emb_a, emb_n, dim=1).sum().item()
                    cnt_v += a.size(0)
            val_loss = tot_val / len(val_loader)
            gap_val = (sum_pos_v / cnt_v) - (sum_neg_v / cnt_v)
            data["losses"]["val"].append(val_loss)
            data["metrics"]["gap_train"].append(gap_train)
            data["metrics"]["gap_val"].append(gap_val)

            # retrieval accuracy
            with torch.no_grad():
                emb_all = model(encoded)
                emb_norm = F.normalize(emb_all, dim=1)
                sims = emb_norm @ emb_norm.T

                def compute_acc(idxs):
                    c = 0
                    for i in idxs:
                        sim = sims[i].clone()
                        sim[i] = -1e9
                        top1 = torch.argmax(sim).item()
                        if index_to_gid[top1] == index_to_gid[i]:
                            c += 1
                    return c / len(idxs)

                train_acc = compute_acc(train_indices)
                val_acc = compute_acc(val_indices)
            data["metrics"]["train_acc"].append(train_acc)
            data["metrics"]["val_acc"].append(val_acc)
            print(
                f"{name} Epoch {epoch}: validation_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}, gap_val = {gap_val:.4f}"
            )

        # final predictions
        model.eval()
        with torch.no_grad():
            emb_all = model(encoded)
            emb_norm = F.normalize(emb_all, dim=1)
            sims = emb_norm @ emb_norm.T
            for i in val_indices:
                sim = sims[i].clone()
                sim[i] = -1e9
                top1 = torch.argmax(sim).item()
                data["predictions"].append(index_to_gid[top1])
                data["ground_truth"].append(index_to_gid[i])

        experiment_data[name]["synthetic"][E] = data
        print(
            f"Finished {name}, E={E}: final val_acc={data['metrics']['val_acc'][-1]:.4f}"
        )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
