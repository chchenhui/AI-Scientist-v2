import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic dataset creation
np.random.seed(0)
torch.manual_seed(0)
N, d = 2000, 5
y = np.random.binomial(1, 0.5, size=N)
X_cont = np.random.randn(N, d) + 2 * y.reshape(-1, 1)
spurious_corr = 0.95
rnd = np.random.rand(N)
s = np.where(rnd < spurious_corr, y, 1 - y)
X = np.concatenate([X_cont, s.reshape(-1, 1)], axis=1)

# Train/val/test split
idxs = np.arange(N)
np.random.shuffle(idxs)
train_idx, val_idx, test_idx = idxs[:1000], idxs[1000:1500], idxs[1500:]

# Normalize
mean = X[train_idx, :d].mean(axis=0)
std = X[train_idx, :d].std(axis=0) + 1e-6
X_norm = X.copy()
X_norm[:, :d] = (X_norm[:, :d] - mean) / std

# Prepare splits
splits = {
    "train": (X_norm[train_idx], y[train_idx], s[train_idx]),
    "val": (X_norm[val_idx], y[val_idx], s[val_idx]),
    "test": (X_norm[test_idx], y[test_idx], s[test_idx]),
}


class SyntheticDataset(Dataset):
    def __init__(self, X, y, g):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.g = torch.tensor(g, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "features": self.X[idx],
            "label": self.y[idx],
            "group": self.g[idx],
            "idx": idx,
        }


# DataLoaders
train_ds = SyntheticDataset(*splits["train"])
val_ds = SyntheticDataset(*splits["val"])
test_ds = SyntheticDataset(*splits["test"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
cluster_loader = DataLoader(train_ds, batch_size=1, shuffle=False)


# K-means helper
def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    init_idxs = rng.choice(N, n_clusters, replace=False)
    centroids = X[init_idxs].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                centroids[k] = pts.mean(axis=0)
            else:
                centroids[k] = X[rng.randint(N)]
    return labels


# Model
class MLP(nn.Module):
    def __init__(self, inp_dim, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU()
        )
        self.fc = nn.Linear(hid, 2)

    def forward(self, x):
        return self.fc(self.net(x))


# Evaluation
criterion = nn.CrossEntropyLoss(reduction="none")


def evaluate(loader, model):
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
            for g in [0, 1]:
                mask = grp == g
                total[g] += mask.sum().item()
                if mask.sum().item() > 0:
                    correct[g] += (preds[mask] == yb[mask]).sum().item()
    wg = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in [0, 1])
    return loss_sum, wg


# Hyperparameter tuning over momentum
momentums = [0.5, 0.9, 0.99]
warmup_epochs, train_epochs = 1, 5
total_epochs = warmup_epochs + train_epochs

experiment_data = {
    "momentum_sweep": {
        "synthetic": {
            "momentum_values": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for m in momentums:
    print(f"\n>>> Running momentum = {m}")
    experiment_data["momentum_sweep"]["synthetic"]["momentum_values"].append(m)
    # init model, optimizer
    model = MLP(d + 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=m)
    sample_weights = None

    # histories
    train_wg_hist, val_wg_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            x, yb, idxb = batch["features"], batch["label"], batch["idx"]
            out = model(x)
            losses = criterion(out, yb)
            if epoch >= warmup_epochs and sample_weights is not None:
                sw = sample_weights[idxb]
                loss = (losses * sw).mean()
            else:
                loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # clustering step after warmup
        if epoch == warmup_epochs - 1:
            model.eval()
            grads = []
            for sample in cluster_loader:
                batch = {
                    k: v.to(device) for k, v in sample.items() if torch.is_tensor(v)
                }
                optimizer.zero_grad()
                out = model(batch["features"])
                loss_i = criterion(out, batch["label"]).mean()
                loss_i.backward()
                g = model.fc.weight.grad.detach().cpu().view(-1).numpy()
                grads.append(g)
            grads = np.stack(grads)
            labels = kmeans_np(grads, n_clusters=2, n_iters=10)
            counts = np.bincount(labels, minlength=2)
            sw_arr = np.array([1.0 / counts[l] for l in labels], dtype=np.float32)
            sample_weights = torch.tensor(sw_arr, device=device)

        # evaluate
        tr_loss, tr_wg = evaluate(train_loader, model)
        val_loss, val_wg = evaluate(val_loader, model)
        train_wg_hist.append(tr_wg)
        val_wg_hist.append(val_wg)
        train_loss_hist.append(tr_loss / len(train_ds))
        val_loss_hist.append(val_loss / len(val_ds))
        print(
            f"momentum={m} epoch={epoch} val_loss={val_loss/len(val_ds):.4f} val_wg={val_wg:.4f}"
        )

    # save histories
    experiment_data["momentum_sweep"]["synthetic"]["metrics"]["train"].append(
        train_wg_hist
    )
    experiment_data["momentum_sweep"]["synthetic"]["metrics"]["val"].append(val_wg_hist)
    experiment_data["momentum_sweep"]["synthetic"]["losses"]["train"].append(
        train_loss_hist
    )
    experiment_data["momentum_sweep"]["synthetic"]["losses"]["val"].append(
        val_loss_hist
    )

    # final test
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(batch["features"])
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            truths.extend(batch["label"].cpu().numpy().tolist())
    experiment_data["momentum_sweep"]["synthetic"]["predictions"].append(
        np.array(preds)
    )
    experiment_data["momentum_sweep"]["synthetic"]["ground_truth"].append(
        np.array(truths)
    )

# Save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
