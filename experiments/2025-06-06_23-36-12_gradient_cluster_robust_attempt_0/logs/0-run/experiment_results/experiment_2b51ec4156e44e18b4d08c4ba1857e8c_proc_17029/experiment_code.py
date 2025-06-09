import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# reproducibility and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
np.random.seed(0)
torch.manual_seed(0)

# synthetic data with spurious feature
N, d = 2000, 5
y = np.random.binomial(1, 0.5, N)
X_cont = np.random.randn(N, d) + 2 * y.reshape(-1, 1)
spurious_corr = 0.95
rnd = np.random.rand(N)
s = np.where(rnd < spurious_corr, y, 1 - y)
X = np.concatenate([X_cont, s.reshape(-1, 1)], axis=1)

# splits
idxs = np.arange(N)
np.random.shuffle(idxs)
train_idx, val_idx, test_idx = idxs[:1000], idxs[1000:1500], idxs[1500:]
mean = X[train_idx, :d].mean(0)
std = X[train_idx, :d].std(0) + 1e-6
X_norm = X.copy()
X_norm[:, :d] = (X_norm[:, :d] - mean) / std
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


# data loaders
train_ds = SyntheticDataset(*splits["train"])
val_ds = SyntheticDataset(*splits["val"])
test_ds = SyntheticDataset(*splits["test"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)


# model
class MLP(nn.Module):
    def __init__(self, inp, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU()
        )
        self.fc = nn.Linear(hid, 2)

    def forward(self, x):
        return self.fc(self.net(x))


# evaluation
def evaluate(loader, model, crit):
    loss_sum = 0.0
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            out = model(b["features"])
            losses = crit(out, b["label"])
            loss_sum += losses.sum().item()
            preds = out.argmax(1)
            for g in (0, 1):
                mask = b["group"] == g
                total[g] += mask.sum().item()
                if mask.any():
                    correct[g] += (preds[mask] == b["label"][mask]).sum().item()
    wg = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return loss_sum, wg


# hyperparameters
lrs = [1e-4, 1e-3, 1e-2]
warmup_epochs = 1
train_epochs = 5
total_epochs = warmup_epochs + train_epochs

# experiment container
experiment_data = {
    "random_cluster_reweighting": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for lr in lrs:
    print(f"\n=== lr={lr} ===")
    model = MLP(d + 1).to(device)
    crit = nn.CrossEntropyLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sample_weights = None

    m_tr, m_val = [], []
    l_tr, l_val = [], []

    for ep in range(total_epochs):
        model.train()
        for b in train_loader:
            b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            out = model(b["features"])
            losses = crit(out, b["label"])
            if ep >= warmup_epochs and sample_weights is not None:
                loss = (losses * sample_weights[b["idx"]]).mean()
            else:
                loss = losses.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # random cluster reweighting after warmup
        if ep == warmup_epochs - 1:
            Ntr = len(train_ds)
            labs = np.random.randint(0, 2, size=Ntr)
            cnt = np.bincount(labs, minlength=2)
            sw = np.array([1.0 / cnt[l] for l in labs], dtype=np.float32)
            sample_weights = torch.tensor(sw, device=device)

        tr_loss, tr_wg = evaluate(train_loader, model, crit)
        v_loss, v_wg = evaluate(val_loader, model, crit)
        avg_v_loss = v_loss / len(val_ds)
        print(
            f"Epoch {ep}: validation_loss = {avg_v_loss:.4f}, Worst-Group Accuracy = {v_wg:.4f}"
        )
        m_tr.append(tr_wg)
        m_val.append(v_wg)
        l_tr.append(tr_loss / len(train_ds))
        l_val.append(avg_v_loss)

    # test set predictions
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for b in test_loader:
            b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
            out = model(b["features"])
            preds.extend(out.argmax(1).cpu().tolist())
            truths.extend(b["label"].cpu().tolist())

    sd = experiment_data["random_cluster_reweighting"]["synthetic"]
    sd["metrics"]["train"].append(m_tr)
    sd["metrics"]["val"].append(m_val)
    sd["losses"]["train"].append(l_tr)
    sd["losses"]["val"].append(l_val)
    sd["predictions"].append(np.array(preds))
    if len(sd["ground_truth"]) == 0:
        sd["ground_truth"] = np.array(truths)

# convert lists to arrays
sd = experiment_data["random_cluster_reweighting"]["synthetic"]
sd["metrics"]["train"] = np.array(sd["metrics"]["train"])
sd["metrics"]["val"] = np.array(sd["metrics"]["val"])
sd["losses"]["train"] = np.array(sd["losses"]["train"])
sd["losses"]["val"] = np.array(sd["losses"]["val"])
sd["predictions"] = np.stack(sd["predictions"])
# 'ground_truth' is already an array

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
