import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device and random seeds
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

# Train/val/test split
idxs = np.arange(N)
np.random.shuffle(idxs)
train_idx, val_idx, test_idx = idxs[:1000], idxs[1000:1500], idxs[1500:]

# Create normalized and raw splits
mean = X[train_idx, :d].mean(axis=0)
std = X[train_idx, :d].std(axis=0) + 1e-6
X_norm = X.copy()
X_norm[:, :d] = (X_norm[:, :d] - mean) / std
splits_norm = {
    "train": (X_norm[train_idx], y[train_idx], s[train_idx]),
    "val": (X_norm[val_idx], y[val_idx], s[val_idx]),
    "test": (X_norm[test_idx], y[test_idx], s[test_idx]),
}
splits_no_norm = {
    "train": (X[train_idx], y[train_idx], s[train_idx]),
    "val": (X[val_idx], y[val_idx], s[val_idx]),
    "test": (X[test_idx], y[test_idx], s[test_idx]),
}


# Dataset and helper functions
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


def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    init = rng.choice(N, n_clusters, replace=False)
    centroids = X[init].copy()
    labels = np.zeros(N, int)
    for _ in range(n_iters):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            pts = X[labels == k]
            centroids[k] = pts.mean(0) if len(pts) > 0 else X[rng.randint(N)]
    return labels


class MLP(nn.Module):
    def __init__(self, inp_dim, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU()
        )
        self.fc = nn.Linear(hid, 2)

    def forward(self, x):
        return self.fc(self.net(x))


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
                if mask.sum() > 0:
                    correct[g] += (preds[mask] == yb[mask]).sum().item()
    wg_acc = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return loss_sum, wg_acc


# Training hyperparameters
lrs = [1e-4, 1e-3, 1e-2]
warmup_epochs = 1
train_epochs = 5
total_epochs = warmup_epochs + train_epochs


def run_experiment(splits, tag=""):
    # DataLoaders
    Xtr, ytr, str_ = splits["train"]
    Xv, yv, sv = splits["val"]
    Xt, yt, st = splits["test"]
    train_ds = SyntheticDataset(Xtr, ytr, str_)
    val_ds = SyntheticDataset(Xv, yv, sv)
    test_ds = SyntheticDataset(Xt, yt, st)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    cluster_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    # Containers
    metrics_train_all, metrics_val_all = [], []
    losses_train_all, losses_val_all = [], []
    predictions_list = []
    ground_truth = None
    inp_dim = Xtr.shape[1]
    # LR sweep
    for lr in lrs:
        print(f"\n=== {tag} lr={lr} ===")
        model = MLP(inp_dim).to(device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        sample_weights = None
        m_tr, m_val, l_tr, l_val = [], [], [], []
        for epoch in range(total_epochs):
            model.train()
            for batch in train_loader:
                batch = {
                    k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                }
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
            # clustering & reweight
            if epoch == warmup_epochs - 1:
                model.eval()
                grads = []
                for sample in cluster_loader:
                    batch = {
                        k: v.to(device) for k, v in sample.items() if torch.is_tensor(v)
                    }
                    optimizer.zero_grad()
                    out_i = model(batch["features"])
                    loss_i = criterion(out_i, batch["label"]).mean()
                    loss_i.backward()
                    g = model.fc.weight.grad.detach().cpu().view(-1).numpy()
                    grads.append(g)
                grads = np.stack(grads)
                labels = kmeans_np(grads, n_clusters=2, n_iters=10)
                counts = np.bincount(labels, minlength=2)
                sw = np.array([1.0 / counts[lab] for lab in labels], dtype=np.float32)
                sample_weights = torch.tensor(sw, device=device)
            # evaluation
            tr_loss, tr_wg = evaluate(train_loader, model, criterion)
            v_loss, v_wg = evaluate(val_loader, model, criterion)
            print(
                f"{tag} lr={lr} epoch={epoch}: val_loss={v_loss/len(val_ds):.4f}, val_wg={v_wg:.4f}"
            )
            m_tr.append(tr_wg)
            m_val.append(v_wg)
            l_tr.append(tr_loss / len(train_ds))
            l_val.append(v_loss / len(val_ds))
        # collect per-lr
        metrics_train_all.append(m_tr)
        metrics_val_all.append(m_val)
        losses_train_all.append(l_tr)
        losses_val_all.append(l_val)
        # test set predictions
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {
                    k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                }
                out = model(batch["features"])
                preds.extend(out.argmax(1).cpu().tolist())
                truths.extend(batch["label"].cpu().tolist())
        preds = np.array(preds)
        truths = np.array(truths)
        predictions_list.append(preds)
        if ground_truth is None:
            ground_truth = truths
    return {
        "metrics_train": np.array(metrics_train_all),
        "metrics_val": np.array(metrics_val_all),
        "losses_train": np.array(losses_train_all),
        "losses_val": np.array(losses_val_all),
        "predictions": np.stack(predictions_list),
        "ground_truth": ground_truth,
    }


# Run ablation: with vs without normalization
experiment_data = {"NO_FEATURE_NORMALIZATION": {}}
for name, splits in [
    ("synthetic_with_norm", splits_norm),
    ("synthetic_no_norm", splits_no_norm),
]:
    res = run_experiment(splits, tag=name)
    experiment_data["NO_FEATURE_NORMALIZATION"][name] = {
        "metrics": {
            "train": res["metrics_train"],
            "val": res["metrics_val"],
        },
        "losses": {
            "train": res["losses_train"],
            "val": res["losses_val"],
        },
        "predictions": res["predictions"],
        "ground_truth": res["ground_truth"],
    }

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
