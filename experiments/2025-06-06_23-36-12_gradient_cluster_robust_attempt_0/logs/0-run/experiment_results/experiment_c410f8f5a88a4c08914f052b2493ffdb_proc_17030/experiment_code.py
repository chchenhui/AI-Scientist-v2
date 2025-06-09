import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Reproducibility and device
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Synthetic data with spurious feature
N, d = 2000, 5
y = np.random.binomial(1, 0.5, size=N)
X_cont = np.random.randn(N, d) + 2 * y.reshape(-1, 1)
spurious_corr = 0.95
rnd = np.random.rand(N)
s = np.where(rnd < spurious_corr, y, 1 - y)
X = np.concatenate([X_cont, s.reshape(-1, 1)], axis=1)

# Split into train/val/test
idxs = np.arange(N)
np.random.shuffle(idxs)
train_idx, val_idx, test_idx = idxs[:1000], idxs[1000:1500], idxs[1500:]

# Normalize continuous part
mean = X[train_idx, :d].mean(0)
std = X[train_idx, :d].std(0) + 1e-6
X_norm = X.copy()
X_norm[:, :d] = (X_norm[:, :d] - mean) / std


# Dataset & loaders
class SyntheticDataset(Dataset):
    def __init__(self, X, y, g):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.g = torch.tensor(g, dtype=torch.long)
        self.idx = torch.arange(len(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "features": self.X[i],
            "label": self.y[i],
            "group": self.g[i],
            "idx": self.idx[i],
        }


splits = {
    "train": (X_norm[train_idx], y[train_idx], s[train_idx]),
    "val": (X_norm[val_idx], y[val_idx], s[val_idx]),
    "test": (X_norm[test_idx], y[test_idx], s[test_idx]),
}
train_ds = SyntheticDataset(*splits["train"])
val_ds = SyntheticDataset(*splits["val"])
test_ds = SyntheticDataset(*splits["test"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
cluster_loader = DataLoader(train_ds, batch_size=1, shuffle=False)


# k-means for cluster‐based reweighting
def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    init = rng.choice(N, n_clusters, replace=False)
    centroids = X[init].copy()
    labels = np.zeros(N, int)
    for _ in range(n_iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new = dists.argmin(1)
        if np.all(new == labels):
            break
        labels = new
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts):
                centroids[k] = pts.mean(0)
            else:
                centroids[k] = X[rng.randint(N)]
    return labels


# Simple MLP
class MLP(nn.Module):
    def __init__(self, inp, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU()
        )
        self.fc = nn.Linear(hid, 2)

    def forward(self, x):
        return self.fc(self.net(x))


# Evaluate: returns (sum_loss, worst‐group accuracy)
def evaluate(loader, model, criterion):
    loss_sum = 0.0
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            x, yb, grp = batch["features"], batch["label"], batch["group"]
            out = model(x)
            losses = criterion(out, yb)
            loss_sum += losses.sum().item()
            preds = out.argmax(1)
            for g in (0, 1):
                m = grp == g
                total[g] += m.sum().item()
                if m.sum() > 0:
                    correct[g] += (preds[m] == yb[m]).sum().item()
    wg = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return loss_sum, wg


# Hyperparameters
lrs = [1e-4, 1e-3, 1e-2]
warmup_epochs = 1
train_epochs = 5
total_epochs = warmup_epochs + train_epochs

# Prepare experiment_data
experiment_data = {
    "SGD_OPTIMIZER": {
        "synthetic": {
            "learning_rates": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Run ablation with vanilla SGD
for lr in lrs:
    print(f"\n=== SGD lr={lr} ===")
    model = MLP(d + 1).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    sample_weights = None

    m_tr, m_val = [], []
    l_tr, l_val = [], []

    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            x, yb, idxb = batch["features"], batch["label"], batch["idx"]
            out = model(x)
            losses = criterion(out, yb)
            if epoch >= warmup_epochs and sample_weights is not None:
                loss = (losses * sample_weights[idxb]).mean()
            else:
                loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # cluster-based reweighting after warmup
        if epoch == warmup_epochs - 1:
            model.eval()
            grads = []
            for sample in cluster_loader:
                sb = {k: v.to(device) for k, v in sample.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                out_i = model(sb["features"])
                li = criterion(out_i, sb["label"]).mean()
                li.backward()
                g = model.fc.weight.grad.detach().cpu().reshape(-1).numpy()
                grads.append(g)
            grads = np.stack(grads)
            labs = kmeans_np(grads, n_clusters=2, n_iters=10)
            cnt = np.bincount(labs, minlength=2)
            sw = np.array([1.0 / cnt[lab] for lab in labs], np.float32)
            sample_weights = torch.tensor(sw, device=device)

        # evaluate
        tr_loss, tr_wg = evaluate(train_loader, model, criterion)
        v_loss, v_wg = evaluate(val_loader, model, criterion)
        print(f"Epoch {epoch}: validation_loss = {v_loss/len(val_ds):.4f}")
        m_tr.append(tr_wg)
        m_val.append(v_wg)
        l_tr.append(tr_loss / len(train_ds))
        l_val.append(v_loss / len(val_ds))

    # test preds
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            b = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(b["features"])
            preds.extend(out.argmax(1).cpu().tolist())
            truths.extend(b["label"].cpu().tolist())

    # log
    syn = experiment_data["SGD_OPTIMIZER"]["synthetic"]
    syn["learning_rates"].append(lr)
    syn["metrics"]["train"].append(m_tr)
    syn["metrics"]["val"].append(m_val)
    syn["losses"]["train"].append(l_tr)
    syn["losses"]["val"].append(l_val)
    syn["predictions"].append(np.array(preds))
    # BUGFIX: avoid ambiguous truth value by checking list length
    if len(syn["ground_truth"]) == 0:
        syn["ground_truth"] = np.array(truths)

# Convert lists to numpy arrays
syn["learning_rates"] = np.array(syn["learning_rates"])
syn["metrics"]["train"] = np.array(syn["metrics"]["train"])
syn["metrics"]["val"] = np.array(syn["metrics"]["val"])
syn["losses"]["train"] = np.array(syn["losses"]["train"])
syn["losses"]["val"] = np.array(syn["losses"]["val"])
syn["predictions"] = np.stack(syn["predictions"])

# Save data
np.save("experiment_data.npy", experiment_data)
print("\nSaved experiment_data.npy")
