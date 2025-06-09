import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# reproducibility and working dir
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# simple numpy k-means
def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    init_idxs = rng.choice(N, n_clusters, replace=False)
    centroids = X[init_idxs].copy()
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


# evaluation helper (returns avg loss per sample and worst‐group accuracy)
criterion = nn.CrossEntropyLoss(reduction="none")


def evaluate(loader):
    total_loss, total_samples = 0.0, 0
    correct = {}
    counts = {}
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, yb, grp = batch["features"], batch["label"], batch["group"]
            out = model(x)
            losses = criterion(out, yb)
            total_loss += losses.sum().item()
            total_samples += yb.size(0)
            preds = out.argmax(1)
            for g in torch.unique(grp).cpu().numpy().tolist():
                mask = grp == g
                correct[g] = correct.get(g, 0) + int(
                    (preds[mask] == yb[mask]).sum().item()
                )
                counts[g] = counts.get(g, 0) + int(mask.sum().item())
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    wg_acc = min(correct[g] / counts[g] for g in counts) if counts else 0.0
    return avg_loss, wg_acc


# dataset wrapper
class TabularDataset(Dataset):
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


# model
class MLP(nn.Module):
    def __init__(self, inp_dim, hid_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        return self.fc(self.net(x))


# 1) Synthetic data
N, d = 2000, 5
y = np.random.binomial(1, 0.5, size=N)
X_cont = np.random.randn(N, d) + 2 * y.reshape(-1, 1)
sp_corr = 0.95
rnd = np.random.rand(N)
s = np.where(rnd < sp_corr, y, 1 - y)
X = np.concatenate([X_cont, s.reshape(-1, 1)], axis=1)
idxs = np.arange(N)
np.random.shuffle(idxs)
train_idx, val_idx, test_idx = idxs[:1000], idxs[1000:1500], idxs[1500:]
mean, std = X[train_idx, :d].mean(0), X[train_idx, :d].std(0) + 1e-6
X_norm = X.copy()
X_norm[:, :d] = (X_norm[:, :d] - mean) / std
spl_synth = {
    "train": (X_norm[train_idx], y[train_idx], s[train_idx]),
    "val": (X_norm[val_idx], y[val_idx], s[val_idx]),
    "test": (X_norm[test_idx], y[test_idx], s[test_idx]),
}


# 2) HuggingFace image datasets (MNIST & Fashion-MNIST)
def load_hf_image(name):
    ds = load_dataset(name)
    train_val = ds["train"].train_test_split(test_size=0.2, seed=0)

    def to_arr(sub):
        Xl, yl, gl = [], [], []
        for ex in sub:
            img = np.array(ex["image"], dtype=np.float32) / 255.0
            Xl.append(img.flatten())
            yl.append(int(ex["label"]))
            gl.append(0)
        return np.stack(Xl), np.array(yl), np.array(gl)

    return {
        "train": to_arr(train_val["train"]),
        "val": to_arr(train_val["test"]),
        "test": to_arr(ds["test"]),
    }


spl_mnist = load_hf_image("mnist")
spl_fashion = load_hf_image("fashion_mnist")

# aggregate
all_splits = {"synthetic": spl_synth, "mnist": spl_mnist, "fashion_mnist": spl_fashion}

# hyperparameters
hidden_dim = 64
lr = 5e-4
batch_size = 128
warmup_epochs = 5
train_epochs = 15
total_epochs = warmup_epochs + train_epochs

# run experiments
experiment_data = {}
for name, splits in all_splits.items():
    Xtr, ytr, gtr = splits["train"]
    Xval, yval, gval = splits["val"]
    Xte, yte, gte = splits["test"]
    inp_dim = Xtr.shape[1]
    num_classes = int(ytr.max()) + 1

    train_ds = TabularDataset(Xtr, ytr, gtr)
    val_ds = TabularDataset(Xval, yval, gval)
    test_ds = TabularDataset(Xte, yte, gte)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
    cluster_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    model = MLP(inp_dim, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sw = None
    train_losses, val_losses = [], []
    train_wgs, val_wgs = [], []

    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, yb, idxb = batch["features"], batch["label"], batch["idx"]
            out = model(x)
            losses = criterion(out, yb)
            if epoch >= warmup_epochs and sw is not None:
                loss = (losses * sw[idxb]).mean()
            else:
                loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # cluster after warmup
        if epoch == warmup_epochs:
            model.eval()
            grads = []
            for sample in cluster_loader:
                batch = {
                    k: v.to(device)
                    for k, v in sample.items()
                    if isinstance(v, torch.Tensor)
                }
                optimizer.zero_grad()
                out = model(batch["features"])
                li = criterion(out, batch["label"]).mean()
                li.backward()
                grads.append(model.fc.weight.grad.detach().cpu().view(-1).numpy())
            G = np.stack(grads)
            labs = kmeans_np(G, n_clusters=2, n_iters=10)
            cnt = np.bincount(labs, minlength=2)
            sw_arr = np.array([1.0 / cnt[l] for l in labs], dtype=np.float32)
            sw = torch.tensor(sw_arr, device=device)

        # evaluation
        tr_loss, tr_wg = evaluate(train_loader)
        val_loss, val_wg = evaluate(val_loader)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_wgs.append(tr_wg)
        val_wgs.append(val_wg)
        print(
            f"{name} epoch={epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_wg={val_wg:.4f}"
        )

    # test
    model.eval()
    preds, truths = [], []
    for batch in test_loader:
        batch = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        out = model(batch["features"])
        preds.extend(out.argmax(1).cpu().tolist())
        truths.extend(batch["label"].cpu().tolist())

    experiment_data[name] = {
        "metrics": {"train": np.array(train_wgs), "val": np.array(val_wgs)},
        "losses": {"train": np.array(train_losses), "val": np.array(val_losses)},
        "predictions": np.array(preds),
        "ground_truth": np.array(truths),
    }

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
