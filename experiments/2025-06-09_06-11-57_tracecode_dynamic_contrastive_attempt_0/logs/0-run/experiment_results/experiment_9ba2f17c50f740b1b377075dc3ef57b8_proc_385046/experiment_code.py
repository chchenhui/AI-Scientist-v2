import os, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# setup working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# synthetic code and dynamic traces
codes = []
for c in range(1, 11):
    codes += [f"def f(x): return x+{c}", f"def f(x): return {c}+x"]
input_set = np.random.randint(-10, 10, size=100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in input_set))

# group by trace
trace_to_indices = {}
for i, t in enumerate(traces):
    trace_to_indices.setdefault(t, []).append(i)
group_to_indices = {g: idxs for g, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for g, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = g

# char‐level encoding
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi.get(c, 0) for c in s] + [0] * (max_len - len(s)) for s in codes]
)

# train/val split at group level
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_g, val_g = all_gids[:split], all_gids[split:]
train_idx = [i for g in train_g for i in group_to_indices[g]]
val_idx = [i for g in val_g for i in group_to_indices[g]]


# dataset with restricted sampling
class CodeDataset(Dataset):
    def __init__(self, enc, grp2idx, idx2gid, allowed_idx):
        self.enc = enc
        self.grp2idx = grp2idx
        self.idx2gid = idx2gid
        self.allowed = set(allowed_idx)

    def __len__(self):
        return len(self.allowed)

    def __getitem__(self, local_i):
        # map local index to global code index
        idx = list(self.allowed)[local_i]
        anchor = self.enc[idx]
        g = self.idx2gid[idx]
        # positives within same group and same split
        pos_cands = [j for j in self.grp2idx[g] if j in self.allowed and j != idx]
        pos = random.choice(pos_cands) if pos_cands else idx
        # negatives from other groups but within split
        neg_gs = [gid for gid in self.grp2idx if gid != g]
        neg_g = random.choice(neg_gs)
        neg_cands = [j for j in self.grp2idx[neg_g] if j in self.allowed]
        neg = (
            random.choice(neg_cands) if neg_cands else random.choice(list(self.allowed))
        )
        return anchor, self.enc[pos], self.enc[neg]


# create datasets and loaders
train_dataset = CodeDataset(encoded, group_to_indices, index_to_gid, train_idx)
val_dataset = CodeDataset(encoded, group_to_indices, index_to_gid, val_idx)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# encoder model
class CodeEncoder(nn.Module):
    def __init__(self, vs, ed=64, hd=64):
        super().__init__()
        self.embed = nn.Embedding(vs, ed, padding_idx=0)
        self.lstm = nn.LSTM(ed, hd, batch_first=True)

    def forward(self, x):
        emb, _ = self.lstm(self.embed(x))
        return emb[:, -1, :]


# hyperparameter tuning
lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
EPOCHS = 10
experiment_data = {"learning_rate": {}}

for lr in lrs:
    model = CodeEncoder(len(stoi), 64, 64).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    res = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for ep in range(EPOCHS):
        # training
        model.train()
        tr_loss = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            la = loss_fn(model(a), model(p), model(n))
            la.backward()
            optimizer.step()
            tr_loss += la.item()
        tr_loss /= len(train_loader)
        res["losses"]["train"].append(tr_loss)
        # validation loss
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                vl += loss_fn(model(a), model(p), model(n)).item()
        vl /= len(val_loader)
        res["losses"]["val"].append(vl)
        # nearest‐neighbor accuracy
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            norm = F.normalize(emb_all, dim=1)
            sims = norm @ norm.T

            def acc(ix):
                if not ix:
                    return 0
                c = 0
                for i in ix:
                    v = sims[i].clone()
                    v[i] = -float("inf")
                    if index_to_gid[v.argmax().item()] == index_to_gid[i]:
                        c += 1
                return c / len(ix)

            tr_acc, va_acc = acc(train_idx), acc(val_idx)
        res["metrics"]["train"].append(tr_acc)
        res["metrics"]["val"].append(va_acc)
        print(f"lr={lr:.1e} ep={ep} val_loss={vl:.4f} val_acc={va_acc:.4f}")
    # final predictions
    with torch.no_grad():
        emb_all = model(encoded.to(device))
        norm = F.normalize(emb_all, dim=1)
        sims = norm @ norm.T
        for i in val_idx:
            v = sims[i].clone()
            v[i] = -float("inf")
            res["predictions"].append(index_to_gid[v.argmax().item()])
            res["ground_truth"].append(index_to_gid[i])
    experiment_data["learning_rate"][str(lr)] = res

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
