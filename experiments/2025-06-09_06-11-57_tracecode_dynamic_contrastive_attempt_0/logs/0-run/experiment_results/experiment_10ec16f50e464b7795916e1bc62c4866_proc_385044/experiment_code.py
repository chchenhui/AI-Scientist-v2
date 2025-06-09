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
# dynamic traces
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
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = gid
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
    def __init__(
        self, encoded, group_to_indices, index_to_gid, hard_negative_pool_size=1
    ):
        self.encoded = encoded
        self.group_to_indices = group_to_indices
        self.index_to_gid = index_to_gid
        self.pool_size = hard_negative_pool_size

    def __len__(self):
        return len(self.index_to_gid)

    def __getitem__(self, idx):
        anchor = self.encoded[idx]
        gid = self.index_to_gid[idx]
        # positive
        pos = idx
        while pos == idx:
            pos = random.choice(self.group_to_indices[gid])
        # negatives pool
        negs = []
        for _ in range(self.pool_size):
            neg_gid = random.choice([g for g in self.group_to_indices if g != gid])
            neg_idx = random.choice(self.group_to_indices[neg_gid])
            negs.append(self.encoded[neg_idx])
        negs = torch.stack(negs, dim=0)  # [pool_size, max_len]
        return anchor, self.encoded[pos], negs


class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# train/val split by group
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]

# hyperparam grid
pool_sizes = [1, 2, 5, 10]
experiment_data = {
    "hard_negative_pool_size": {
        "synthetic": {
            "pool_sizes": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

EPOCHS = 10
for pool_size in pool_sizes:
    # new dataset, loaders, model, optimizer per run
    dataset = CodeDataset(
        encoded, group_to_indices, index_to_gid, hard_negative_pool_size=pool_size
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=8, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)
    model = CodeEncoder(len(stoi), 64, 64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    losses_train_run, losses_val_run = [], []
    metrics_train_run, metrics_val_run = [], []

    for epoch in range(EPOCHS):
        # training
        model.train()
        train_loss = 0.0
        for a, p, ns in train_loader:
            a, p, ns = a.to(device), p.to(device), ns.to(device)
            emb_a = model(a)
            emb_p = model(p)
            B, K, L = ns.size()
            emb_ns = model(ns.view(-1, L)).view(B, K, -1)
            sim = F.cosine_similarity(emb_a.unsqueeze(1), emb_ns, dim=2)  # [B,K]
            hard_idx = torch.argmax(sim, dim=1)
            emb_n = emb_ns[torch.arange(B), hard_idx]
            loss = loss_fn(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        losses_train_run.append(train_loss)

        # validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for a, p, ns in val_loader:
                a, p, ns = a.to(device), p.to(device), ns.to(device)
                emb_a = model(a)
                emb_p = model(p)
                B, K, L = ns.size()
                emb_ns = model(ns.view(-1, L)).view(B, K, -1)
                sim = F.cosine_similarity(emb_a.unsqueeze(1), emb_ns, dim=2)
                hard_idx = torch.argmax(sim, dim=1)
                emb_n = emb_ns[torch.arange(B), hard_idx]
                val_loss += loss_fn(emb_a, emb_p, emb_n).item()
        val_loss /= len(val_loader)
        losses_val_run.append(val_loss)

        # retrieval accuracy
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            emb_norm = F.normalize(emb_all, dim=1)
            sims = emb_norm @ emb_norm.T

            def compute_acc(indices):
                correct = 0
                for i in indices:
                    sim_i = sims[i].clone()
                    sim_i[i] = -float("inf")
                    top1 = torch.argmax(sim_i).item()
                    if index_to_gid[top1] == index_to_gid[i]:
                        correct += 1
                return correct / len(indices) if indices else 0

            train_acc = compute_acc(train_indices)
            val_acc = compute_acc(val_indices)
        metrics_train_run.append(train_acc)
        metrics_val_run.append(val_acc)

        print(
            f"Pool {pool_size} | Epoch {epoch} | val_loss {val_loss:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}"
        )

    # record run results
    exp = experiment_data["hard_negative_pool_size"]["synthetic"]
    exp["pool_sizes"].append(pool_size)
    exp["losses"]["train"].append(losses_train_run)
    exp["losses"]["val"].append(losses_val_run)
    exp["metrics"]["train"].append(metrics_train_run)
    exp["metrics"]["val"].append(metrics_val_run)

    # final predictions & ground truth
    with torch.no_grad():
        emb_all = model(encoded.to(device))
        emb_norm = F.normalize(emb_all, dim=1)
        sims = emb_norm @ emb_norm.T
        preds, gts = [], []
        for i in val_indices:
            sim_i = sims[i].clone()
            sim_i[i] = -float("inf")
            top1 = torch.argmax(sim_i).item()
            preds.append(index_to_gid[top1])
            gts.append(index_to_gid[i])
    exp["predictions"].append(preds)
    exp["ground_truth"].append(gts)

# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
