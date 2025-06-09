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

# group snippets by trace
trace_to_indices = {}
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: ids for gid, ids in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for idx in idxs:
        index_to_gid[idx] = gid

# encode as char tokens
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = []
for s in codes:
    seq = [stoi[c] for c in s] + [0] * (max_len - len(s))
    encoded.append(seq)
encoded = torch.LongTensor(encoded)


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
# split groups into train/val
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)


# define encoder
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


model = CodeEncoder(len(stoi), 64, 64).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = nn.TripletMarginLoss(margin=1.0)

experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

EPOCHS = 10
for epoch in range(EPOCHS):
    # training
    model.train()
    train_loss = 0.0
    for a, p, n in train_loader:
        a, p, n = a.to(device), p.to(device), n.to(device)
        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)
        loss = loss_fn(emb_a, emb_p, emb_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    experiment_data["synthetic"]["losses"]["train"].append(train_loss)
    # validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for a, p, n in val_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            val_loss += loss_fn(emb_a, emb_p, emb_n).item()
    val_loss /= len(val_loader)
    experiment_data["synthetic"]["losses"]["val"].append(val_loss)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    # retrieval accuracy on train and val
    with torch.no_grad():
        emb_all = model(encoded.to(device))
        emb_norm = F.normalize(emb_all, dim=1)
        sims = emb_norm @ emb_norm.T

        def compute_acc(indices):
            correct = 0
            for i in indices:
                sim = sims[i].clone()
                sim[i] = -float("inf")
                top1 = torch.argmax(sim).item()
                if index_to_gid[top1] == index_to_gid[i]:
                    correct += 1
            return correct / len(indices) if indices else 0

        train_acc = compute_acc(train_indices)
        val_acc = compute_acc(val_indices)
    experiment_data["synthetic"]["metrics"]["train"].append(train_acc)
    experiment_data["synthetic"]["metrics"]["val"].append(val_acc)

# store final predictions and ground truth for val
with torch.no_grad():
    emb_all = model(encoded.to(device))
    emb_norm = F.normalize(emb_all, dim=1)
    sims = emb_norm @ emb_norm.T
    for i in val_indices:
        sim = sims[i].clone()
        sim[i] = -float("inf")
        top1 = torch.argmax(sim).item()
        experiment_data["synthetic"]["predictions"].append(index_to_gid[top1])
        experiment_data["synthetic"]["ground_truth"].append(index_to_gid[i])

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Top-1 Trace-Equivalence Retrieval Accuracy: {val_acc:.4f}")
