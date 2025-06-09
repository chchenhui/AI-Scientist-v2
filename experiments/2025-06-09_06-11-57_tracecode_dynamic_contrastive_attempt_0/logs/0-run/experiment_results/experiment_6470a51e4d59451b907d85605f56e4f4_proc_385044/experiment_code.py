import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# synthetic code and traces
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
trace_to_indices = {}
for idx, tr in enumerate(traces):
    trace_to_indices.setdefault(tr, []).append(idx)
group_to_indices = {g: ids for g, ids in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for g, ids in group_to_indices.items():
    for i in ids:
        index_to_gid[i] = g

# token encoding
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi[c] for c in s] + [0] * (max_len - len(s)) for s in codes]
)


# Dataset
class CodeDataset(Dataset):
    def __init__(self, enc, g2i, i2g):
        self.encoded = enc
        self.group_to_indices = g2i
        self.index_to_gid = i2g

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
# train/val split by groups
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)


# encoder
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# hyperparam sweep
embed_dims = [32, 64, 128, 256]
EPOCHS = 10
experiment_data = {
    "embed_dim": {
        "synthetic": {
            "values": embed_dims,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for ed in embed_dims:
    model = CodeEncoder(len(stoi), embed_dim=ed, hidden=64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    run_metrics_train, run_metrics_val = [], []
    run_losses_train, run_losses_val = [], []

    for epoch in range(EPOCHS):
        # train
        model.train()
        tot_loss = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            emb_a, emb_p, emb_n = model(a), model(p), model(n)
            loss = loss_fn(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        avg_train_loss = tot_loss / len(train_loader)
        run_losses_train.append(avg_train_loss)

        # val loss
        model.eval()
        tot_vloss = 0.0
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                emb_a, emb_p, emb_n = model(a), model(p), model(n)
                tot_vloss += loss_fn(emb_a, emb_p, emb_n).item()
        run_losses_val.append(tot_vloss / len(val_loader))

        # retrieval acc
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            emb_norm = F.normalize(emb_all, dim=1)
            sims = emb_norm @ emb_norm.T

            def comp_acc(indices):
                c = 0
                for i in indices:
                    sim = sims[i].clone()
                    sim[i] = -float("inf")
                    top1 = torch.argmax(sim).item()
                    if index_to_gid[top1] == index_to_gid[i]:
                        c += 1
                return c / len(indices)

            run_metrics_train.append(comp_acc(train_indices))
            run_metrics_val.append(comp_acc(val_indices))

    # final preds and gt
    preds, gts = [], []
    with torch.no_grad():
        emb_all = model(encoded.to(device))
        emb_norm = F.normalize(emb_all, dim=1)
        sims = emb_norm @ emb_norm.T
        for i in val_indices:
            sim = sims[i].clone()
            sim[i] = -float("inf")
            top1 = torch.argmax(sim).item()
            preds.append(index_to_gid[top1])
            gts.append(index_to_gid[i])

    # collect
    d = experiment_data["embed_dim"]["synthetic"]
    d["metrics"]["train"].append(run_metrics_train)
    d["metrics"]["val"].append(run_metrics_val)
    d["losses"]["train"].append(run_losses_train)
    d["losses"]["val"].append(run_losses_val)
    d["predictions"].append(preds)
    d["ground_truth"].append(gts)
    print(f"Embed_dim={ed} | final val acc={run_metrics_val[-1]:.4f}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
