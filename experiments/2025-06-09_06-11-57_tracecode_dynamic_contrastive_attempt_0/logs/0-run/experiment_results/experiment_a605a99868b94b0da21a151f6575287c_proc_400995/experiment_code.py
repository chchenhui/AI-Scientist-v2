import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# synthetic data
codes = []
for c in range(1, 11):
    codes.append(f"def f(x): return x+{c}")
    codes.append(f"def f(x): return {c}+x")
inputs = np.random.randint(-10, 10, size=100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in inputs))
trace_to_indices = {}
for idx, t in enumerate(traces):
    trace_to_indices.setdefault(t, []).append(idx)
group_to_indices = {g: idxs for g, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = gid

# encode
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi[c] for c in s] + [0] * (max_len - len(s)) for s in codes]
)


# dataset
class CodeDataset(Dataset):
    def __init__(self, encoded, group_to_indices, index_to_gid):
        self.enc = encoded
        self.g2i = group_to_indices
        self.i2g = index_to_gid

    def __len__(self):
        return len(self.i2g)

    def __getitem__(self, idx):
        a = self.enc[idx]
        gid = self.i2g[idx]
        pos = idx
        while pos == idx:
            pos = random.choice(self.g2i[gid])
        neg_gid = random.choice([g for g in self.g2i if g != gid])
        neg = random.choice(self.g2i[neg_gid])
        return a, self.enc[pos], self.enc[neg]


dataset = CodeDataset(encoded, group_to_indices, index_to_gid)
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_g, val_g = all_gids[:split], all_gids[split:]
train_idx = [i for g in train_g for i in group_to_indices[g]]
val_idx = [i for g in val_g for i in group_to_indices[g]]
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=8, shuffle=False)


# model
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x, _ = self.embed(x), None
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# NT-Xent
def nt_xent_loss(z1, z2, tau=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z, z.T) / tau
    mask = ~torch.eye(2 * N, device=sim.device, dtype=torch.bool)
    sim_masked = sim.masked_fill(~mask, -1e9)
    log_prob = F.log_softmax(sim_masked, dim=1)
    pos_idx = (
        torch.arange(N, 2 * N, device=sim.device).tolist()
        + torch.arange(0, N, device=sim.device).tolist()
    )
    pos_idx = torch.tensor(pos_idx, device=sim.device)
    loss = -log_prob[torch.arange(2 * N, device=sim.device), pos_idx].mean()
    return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_kwargs = dict(vocab_size=len(stoi), embed_dim=64, hidden=64)
EPOCH_LIST = [10, 30, 50]
experiment_data = {"triplet": {"synthetic": {}}, "contrastive": {"synthetic": {}}}

for ablation in ["triplet", "contrastive"]:
    for E in EPOCH_LIST:
        model = CodeEncoder(**model_kwargs).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0) if ablation == "triplet" else None
        data = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        for epoch in range(E):
            model.train()
            tot_tr = 0
            for a, p, n in train_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                za, zp = model(a), model(p)
                if ablation == "triplet":
                    zn = model(n)
                    loss = loss_fn(za, zp, zn)
                else:
                    loss = nt_xent_loss(za, zp, tau=0.5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_tr += loss.item()
            avg_tr = tot_tr / len(train_loader)
            data["losses"]["train"].append(avg_tr)

            model.eval()
            tot_val = 0
            with torch.no_grad():
                for a, p, n in val_loader:
                    a, p = a.to(device), p.to(device)
                    if ablation == "triplet":
                        loss = loss_fn(model(a), model(p), model(n.to(device)))
                    else:
                        loss = nt_xent_loss(model(a), model(p), tau=0.5)
                    tot_val += loss.item()
            avg_val = tot_val / len(val_loader)
            data["losses"]["val"].append(avg_val)

            # retrieval
            with torch.no_grad():
                emb = model(encoded.to(device))
                emb = F.normalize(emb, dim=1)
                sims = emb @ emb.T

                def acc(idxs):
                    c = 0
                    for i in idxs:
                        row = sims[i].clone()
                        row[i] = -1e9
                        if index_to_gid[int(row.argmax())] == index_to_gid[i]:
                            c += 1
                    return c / len(idxs)

                data["metrics"]["train"].append(acc(train_idx))
                data["metrics"]["val"].append(acc(val_idx))

        # final preds
        model.eval()
        with torch.no_grad():
            emb = F.normalize(model(encoded.to(device)), dim=1)
            sims = emb @ emb.T
            for i in val_idx:
                row = sims[i].clone()
                row[i] = -1e9
                pred = int(row.argmax())
                data["predictions"].append(index_to_gid[pred])
                data["ground_truth"].append(index_to_gid[i])

        experiment_data[ablation]["synthetic"][E] = data
        print(f"{ablation} E={E} val_acc={data['metrics']['val'][-1]:.4f}")

np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
