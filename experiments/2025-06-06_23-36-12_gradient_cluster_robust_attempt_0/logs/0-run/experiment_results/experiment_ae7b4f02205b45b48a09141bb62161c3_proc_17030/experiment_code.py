import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set device and seeds
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

# Normalize cont features
mean = X[train_idx, :d].mean(0)
std = X[train_idx, :d].std(0) + 1e-6
X_norm = X.copy()
X_norm[:, :d] = (X_norm[:, :d] - mean) / std

# Splits
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

    def __getitem__(self, i):
        return {"features": self.X[i], "label": self.y[i], "group": self.g[i], "idx": i}


# DataLoaders
train_ds = SyntheticDataset(*splits["train"])
val_ds = SyntheticDataset(*splits["val"])
test_ds = SyntheticDataset(*splits["test"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)


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


# Eval helper
def evaluate(loader, model, criterion):
    loss_sum = 0.0
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items() if torch.is_tensor(v)}
            out = model(b["features"])
            l = criterion(out, b["label"])
            loss_sum += l.sum().item()
            preds = out.argmax(1)
            for g in (0, 1):
                m = b["group"] == g
                total[g] += m.sum().item()
                if m.sum() > 0:
                    correct[g] += (preds[m] == b["label"][m]).sum().item()
    wg = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return loss_sum, wg


# Hyperparameters
lrs = [1e-4, 1e-3, 1e-2]
warmup_epochs = 1
train_epochs = 5
total_epochs = warmup_epochs + train_epochs

# Containers
metrics_train_all, metrics_val_all = [], []
losses_train_all, losses_val_all = [], []
predictions_list = []
ground_truth = None
sample_weights_list = []

# Ablation: loss-based reweighting
for lr in lrs:
    print(f"\nLR={lr}")
    model = MLP(d + 1).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sample_weights = None

    m_tr, m_val = [], []
    l_tr, l_val = [], []

    for epoch in range(total_epochs):
        model.train()
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items() if torch.is_tensor(v)}
            out = model(b["features"])
            losses = criterion(out, b["label"])
            if epoch >= warmup_epochs and sample_weights is not None:
                loss = (losses * sample_weights[b["idx"]]).mean()
            else:
                loss = losses.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        # compute loss-based weights
        if epoch == warmup_epochs - 1:
            model.eval()
            all_loss = torch.zeros(len(train_ds), dtype=torch.float32, device=device)
            with torch.no_grad():
                for b in train_loader:
                    b = {k: v.to(device) for k, v in b.items() if torch.is_tensor(v)}
                    out = model(b["features"])
                    l = criterion(out, b["label"])
                    all_loss[b["idx"]] = l
            w = all_loss.cpu().numpy()
            w = w / w.sum()
            sample_weights = torch.tensor(w, dtype=torch.float32, device=device)
            sample_weights_list.append(w)

        # evaluate
        tr_loss, tr_wg = evaluate(train_loader, model, criterion)
        v_loss, v_wg = evaluate(val_loader, model, criterion)
        print(f"Epoch {epoch}: val_loss={v_loss/len(val_ds):.4f}, val_wg={v_wg:.4f}")
        m_tr.append(tr_wg)
        m_val.append(v_wg)
        l_tr.append(tr_loss / len(train_ds))
        l_val.append(v_loss / len(val_ds))

    # test preds
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for b in test_loader:
            b = {k: v.to(device) for k, v in b.items() if torch.is_tensor(v)}
            out = model(b["features"])
            preds.extend(out.argmax(1).cpu().tolist())
            truths.extend(b["label"].cpu().tolist())
    preds = np.array(preds)
    truths = np.array(truths)
    predictions_list.append(preds)
    if ground_truth is None:
        ground_truth = truths

    metrics_train_all.append(m_tr)
    metrics_val_all.append(m_val)
    losses_train_all.append(l_tr)
    losses_val_all.append(l_val)

# Save experiment data
experiment_data = {
    "LOSS_BASED_SAMPLE_WEIGHTING": {
        "synthetic": {
            "metrics": {
                "train": np.array(metrics_train_all),
                "val": np.array(metrics_val_all),
            },
            "losses": {
                "train": np.array(losses_train_all),
                "val": np.array(losses_val_all),
            },
            "predictions": np.stack(predictions_list),
            "ground_truth": ground_truth,
            "sample_weights": np.stack(sample_weights_list),
        }
    }
}

np.save("experiment_data.npy", experiment_data)
