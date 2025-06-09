import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic dataset with spurious feature
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
splits = {}
splits["train"] = (X_norm[train_idx], y[train_idx], s[train_idx])
splits["val"] = (X_norm[val_idx], y[val_idx], s[val_idx])
splits["test"] = (X_norm[test_idx], y[test_idx], s[test_idx])


# Dataset class
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


# Loaders
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
    idxs = rng.choice(N, n_clusters, replace=False)
    centroids = X[idxs].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
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


def evaluate(loader):
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
                if mask.sum() > 0:
                    correct[g] += (preds[mask] == yb[mask]).sum().item()
    wg_acc = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in [0, 1])
    return loss_sum, wg_acc


# Hyperparameter tuning for warmup_epochs
warmup_list = [1, 2, 3, 5]
train_epochs = 5
experiment_data = {
    "warmup_epochs": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": splits["test"][1],
            "warmup_values": warmup_list,
        }
    }
}

# Loop over hyperparams
for w in warmup_list:
    # re-seed for reproducible init
    torch.manual_seed(0)
    model = MLP(d + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample_weights = None

    run_train_losses = []
    run_val_losses = []
    run_train_metrics = []
    run_val_metrics = []

    total_epochs = w + train_epochs
    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            x, yb, idxb = batch["features"], batch["label"], batch["idx"]
            out = model(x)
            losses = criterion(out, yb)
            if epoch >= w and sample_weights is not None:
                sw = sample_weights[idxb]
                loss = (losses * sw).mean()
            else:
                loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # cluster and compute weights
        if epoch == w - 1:
            model.eval()
            grads = []
            for sample in cluster_loader:
                batch = {
                    k: v.to(device) for k, v in sample.items() if torch.is_tensor(v)
                }
                optimizer.zero_grad()
                out = model(batch["features"])
                l_i = criterion(out, batch["label"]).mean()
                l_i.backward()
                g = model.fc.weight.grad.detach().cpu().view(-1).numpy()
                grads.append(g)
            grads = np.stack(grads)
            labels = kmeans_np(grads, n_clusters=2, n_iters=10)
            counts = np.bincount(labels, minlength=2)
            sw_arr = np.array([1.0 / counts[lab] for lab in labels], dtype=np.float32)
            sample_weights = torch.tensor(sw_arr, device=device)

        # evaluate
        tr_l, tr_m = evaluate(train_loader)
        vl_l, vl_m = evaluate(val_loader)
        run_train_losses.append(tr_l / len(train_ds))
        run_val_losses.append(vl_l / len(val_ds))
        run_train_metrics.append(tr_m)
        run_val_metrics.append(vl_m)
        print(
            f"warmup={w} epoch={epoch}: val_loss={vl_l/len(val_ds):.4f} wg_val={vl_m:.4f}"
        )

    # test predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(batch["features"])
            preds.extend(out.argmax(1).cpu().numpy().tolist())

    # store results
    exp = experiment_data["warmup_epochs"]["synthetic"]
    exp["losses"]["train"].append(run_train_losses)
    exp["losses"]["val"].append(run_val_losses)
    exp["metrics"]["train"].append(run_train_metrics)
    exp["metrics"]["val"].append(run_val_metrics)
    exp["predictions"].append(np.array(preds))

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
