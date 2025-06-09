import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)

# Synthetic data with spurious feature
N, d = 2000, 5
y = np.random.binomial(1, 0.5, size=N)
X_cont = np.random.randn(N, d) + 2 * y.reshape(-1, 1)
spurious_corr = 0.95
rnd = np.random.rand(N)
s = np.where(rnd < spurious_corr, y, 1 - y)
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


# k-means on CPU
def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    init_idxs = rng.choice(N, n_clusters, replace=False)
    centroids = X[init_idxs].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
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


# MLP model
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
                mask = grp == g
                total[g] += mask.sum().item()
                if mask.sum().item() > 0:
                    correct[g] += (preds[mask] == yb[mask]).sum().item()
    wg_acc = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return loss_sum, wg_acc


# Hyperparameters
lrs = [1e-4, 1e-3, 1e-2]
warmup_epochs = 1
train_epochs = 5
total_epochs = warmup_epochs + train_epochs

# Prepare experiment data container
experiment_data = {
    "INPUT_FEATURE_CLUSTER_REWEIGHTING": {
        "synthetic": {
            "lrs": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for lr in lrs:
    print(f"\n=== LR={lr} ===")
    model = MLP(d + 1).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

        # After warmup: cluster input features and set weights
        if epoch == warmup_epochs - 1:
            X_train = splits["train"][0]  # numpy array
            labels = kmeans_np(X_train, n_clusters=2, n_iters=10)
            counts = np.bincount(labels, minlength=2)
            sw_arr = np.array([1.0 / counts[lab] for lab in labels], dtype=np.float32)
            sample_weights = torch.tensor(sw_arr, device=device)

        # Evaluate
        tr_loss, tr_wg = evaluate(train_loader, model, criterion)
        v_loss, v_wg = evaluate(val_loader, model, criterion)
        print(f"Epoch {epoch}: val_loss={v_loss/len(val_ds):.4f}, val_wg={v_wg:.4f}")
        m_tr.append(tr_wg)
        m_val.append(v_wg)
        l_tr.append(tr_loss / len(train_ds))
        l_val.append(v_loss / len(val_ds))

    # Final test predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(batch["features"])
            preds.extend(out.argmax(1).cpu().tolist())

    # Collect into experiment_data
    exp = experiment_data["INPUT_FEATURE_CLUSTER_REWEIGHTING"]["synthetic"]
    exp["lrs"].append(lr)
    exp["metrics"]["train"].append(m_tr)
    exp["metrics"]["val"].append(m_val)
    exp["losses"]["train"].append(l_tr)
    exp["losses"]["val"].append(l_val)
    exp["predictions"].append(preds)
    if not exp["ground_truth"]:
        exp["ground_truth"] = splits["test"][1].tolist()

# Convert lists to numpy arrays and save
syn = experiment_data["INPUT_FEATURE_CLUSTER_REWEIGHTING"]["synthetic"]
syn["lrs"] = np.array(syn["lrs"])
syn["metrics"]["train"] = np.array(syn["metrics"]["train"])
syn["metrics"]["val"] = np.array(syn["metrics"]["val"])
syn["losses"]["train"] = np.array(syn["losses"]["train"])
syn["losses"]["val"] = np.array(syn["losses"]["val"])
syn["predictions"] = np.array(syn["predictions"])
syn["ground_truth"] = np.array(syn["ground_truth"])

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
