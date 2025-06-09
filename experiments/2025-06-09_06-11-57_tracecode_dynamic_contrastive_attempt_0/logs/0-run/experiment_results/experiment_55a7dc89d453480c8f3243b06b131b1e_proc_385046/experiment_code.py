import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic code snippets
codes = []
for c in range(1, 11):
    codes.append(f"def f(x): return x+{c}")
    codes.append(f"def f(x): return {c}+x")

# generate dynamic traces
input_set = np.random.randint(-10, 10, size=100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in input_set))

# group by trace
trace_to_indices = {}
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [0] * len(codes)
for gid, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = gid

# encode as char tokens
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi[c] for c in s] + [0] * (max_len - len(s)) for s in codes]
)


# dataset and loaders
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


# encoder model
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# helper for retrieval accuracy
def compute_acc(sims, indices):
    correct = 0
    for i in indices:
        v = sims[i].clone()
        v[i] = -float("inf")
        top1 = torch.argmax(v).item()
        if index_to_gid[top1] == index_to_gid[i]:
            correct += 1
    return correct / len(indices) if indices else 0


# hyperparameter sweep for weight_decay
weight_decay_values = [0, 1e-5, 1e-4, 1e-3, 1e-2]
EPOCHS = 10
experiment_data = {
    "weight_decay_sweep": {
        "synthetic": {
            "params": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for wd in weight_decay_values:
    print(f"\n--- Training with weight_decay = {wd} ---")
    experiment_data["weight_decay_sweep"]["synthetic"]["params"].append(wd)
    # init model, optimizer, loss
    model = CodeEncoder(len(stoi), 64, 64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    # storage per run
    run_train_losses, run_val_losses = [], []
    run_train_acc, run_val_acc = [], []
    # training loop
    for epoch in range(EPOCHS):
        model.train()
        tloss = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            emb_a, emb_p, emb_n = model(a), model(p), model(n)
            loss = loss_fn(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        tloss /= len(train_loader)
        run_train_losses.append(tloss)
        # validation loss
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                vloss += loss_fn(model(a), model(p), model(n)).item()
        vloss /= len(val_loader)
        run_val_losses.append(vloss)
        # retrieval metrics
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            emb_norm = F.normalize(emb_all, dim=1)
            sims = emb_norm @ emb_norm.T
            tra = compute_acc(sims, train_indices)
            va = compute_acc(sims, val_indices)
        run_train_acc.append(tra)
        run_val_acc.append(va)
        print(
            f"WD {wd} Epoch {epoch}: train_loss={tloss:.4f}, val_loss={vloss:.4f}, train_acc={tra:.4f}, val_acc={va:.4f}"
        )
    # record run data
    sd = experiment_data["weight_decay_sweep"]["synthetic"]
    sd["losses"]["train"].append(run_train_losses)
    sd["losses"]["val"].append(run_val_losses)
    sd["metrics"]["train"].append(run_train_acc)
    sd["metrics"]["val"].append(run_val_acc)
    # predictions and ground truth
    preds, gts = [], []
    with torch.no_grad():
        emb_all = model(encoded.to(device))
        emb_norm = F.normalize(emb_all, dim=1)
        sims = emb_norm @ emb_norm.T
        for i in val_indices:
            s = sims[i].clone()
            s[i] = -float("inf")
            top1 = torch.argmax(s).item()
            preds.append(index_to_gid[top1])
            gts.append(index_to_gid[i])
    sd["predictions"].append(preds)
    sd["ground_truth"].append(gts)

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy with weight_decay results.")
