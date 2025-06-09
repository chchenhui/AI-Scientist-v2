import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed
np.random.seed(0)
torch.manual_seed(0)

# Create synthetic dataset
N, d = 2000, 5
y = np.random.binomial(1, 0.5, size=N)
X_cont = np.random.randn(N, d) + 2 * y.reshape(-1, 1)
s = np.where(np.random.rand(N) < 0.95, y, 1 - y)
X = np.concatenate([X_cont, s.reshape(-1, 1)], axis=1)

# Split indices
idxs = np.arange(N)
np.random.shuffle(idxs)
train_idx, val_idx, test_idx = idxs[:1000], idxs[1000:1500], idxs[1500:]

# Normalize continuous features
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


# Dataset and DataLoader
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


train_ds = SyntheticDataset(*splits["train"])
val_ds = SyntheticDataset(*splits["val"])
test_ds = SyntheticDataset(*splits["test"])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
cluster_loader = DataLoader(train_ds, batch_size=1, shuffle=False)


# Simple NumPy k-means
def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    centroids = X[rng.choice(N, n_clusters, replace=False)].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
        new_labels = dists.argmin(1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                centroids[k] = pts.mean(0)
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


# Evaluation helper
def evaluate(loader, model, criterion):
    model.eval()
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
    wg_acc = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in [0, 1])
    return loss_sum, wg_acc


# Hyperparameter tuning: number of clusters
param_grid = [2, 3, 4, 5]
warmup_epochs = 1
train_epochs = 5
total_epochs = warmup_epochs + train_epochs

experiment_data = {
    "n_clusters_tuning": {
        "synthetic": {
            "n_clusters": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for nc in param_grid:
    print(f"Running n_clusters = {nc}")
    # re-init model & optimizer
    model = MLP(d + 1).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample_weights = None

    run_m_train, run_m_val = [], []
    run_l_train, run_l_val = [], []

    # Training loop
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

        # cluster gradients after warmup
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
                grads.append(model.fc.weight.grad.detach().cpu().view(-1).numpy())
            grads = np.stack(grads)
            labels = kmeans_np(grads, n_clusters=nc, n_iters=10)
            counts = np.bincount(labels, minlength=nc)
            sw_arr = np.array([1.0 / counts[l] for l in labels], dtype=np.float32)
            sample_weights = torch.tensor(sw_arr, device=device)

        # evaluation
        tr_loss, tr_wg = evaluate(train_loader, model, criterion)
        val_loss, val_wg = evaluate(val_loader, model, criterion)
        print(f" n={nc} Epoch {epoch}: val_wg={val_wg:.4f}")
        run_l_train.append(tr_loss / len(train_ds))
        run_l_val.append(val_loss / len(val_ds))
        run_m_train.append(tr_wg)
        run_m_val.append(val_wg)

    # final test
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(batch["features"])
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            truths.extend(batch["label"].cpu().numpy().tolist())

    # record results
    ed = experiment_data["n_clusters_tuning"]["synthetic"]
    ed["n_clusters"].append(nc)
    ed["losses"]["train"].append(run_l_train)
    ed["losses"]["val"].append(run_l_val)
    ed["metrics"]["train"].append(run_m_train)
    ed["metrics"]["val"].append(run_m_val)
    ed["predictions"].append(np.array(preds))
    ed["ground_truth"].append(np.array(truths))

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
