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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# build synthetic code snippets and traces
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
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = gid

# encode snippets as character tokens
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi[c] for c in s] + [0] * (max_len - len(s)) for s in codes]
)


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


class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# hyperparameter sweep over batch sizes
batch_sizes = [4, 16, 32]
EPOCHS = 10
experiment_data = {"batch_size": {"synthetic": []}}

for bs in batch_sizes:
    print(f"\n=== Training with batch size = {bs} ===")
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=bs, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=bs, shuffle=False)
    model = CodeEncoder(len(stoi), 64, 64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        tloss = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            loss = loss_fn(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        tloss /= len(train_loader)
        train_losses.append(tloss)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                emb_a = model(a)
                emb_p = model(p)
                emb_n = model(n)
                vloss += loss_fn(emb_a, emb_p, emb_n).item()
        vloss /= len(val_loader)
        val_losses.append(vloss)
        print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={tloss:.4f}, val_loss={vloss:.4f}")

        # retrieval accuracy
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            emb_norm = F.normalize(emb_all, dim=1)
            sims = emb_norm @ emb_norm.T

            def comp_acc(idxs):
                correct = 0
                for i in idxs:
                    row = sims[i].clone()
                    row[i] = -float("inf")
                    if index_to_gid[int(torch.argmax(row))] == index_to_gid[i]:
                        correct += 1
                return correct / len(idxs)

            train_accs.append(comp_acc(train_indices))
            val_accs.append(comp_acc(val_indices))
        print(f"  acc -> train: {train_accs[-1]:.4f}, val: {val_accs[-1]:.4f}")

    # final predictions & ground truth
    predictions, ground_truth = [], []
    with torch.no_grad():
        emb_all = model(encoded.to(device))
        emb_norm = F.normalize(emb_all, dim=1)
        sims = emb_norm @ emb_norm.T
        for i in val_indices:
            row = sims[i].clone()
            row[i] = -float("inf")
            predictions.append(index_to_gid[int(torch.argmax(row))])
            ground_truth.append(index_to_gid[i])

    # store results for this batch size
    experiment_data["batch_size"]["synthetic"].append(
        {
            "batch_size": bs,
            "metrics": {"train": train_accs, "val": val_accs},
            "losses": {"train": train_losses, "val": val_losses},
            "predictions": predictions,
            "ground_truth": ground_truth,
        }
    )

# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment data to:", os.path.join(working_dir, "experiment_data.npy"))
