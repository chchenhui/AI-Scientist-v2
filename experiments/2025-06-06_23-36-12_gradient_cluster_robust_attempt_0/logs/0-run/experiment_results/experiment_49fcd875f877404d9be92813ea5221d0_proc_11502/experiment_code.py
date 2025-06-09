import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import random

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# K-means on NumPy
def kmeans_np(X, n_clusters=2, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    centroids = X[rng.choice(N, n_clusters, replace=False)].copy()
    labels = np.zeros(N, int)
    for _ in range(n_iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            pts = X[labels == k]
            centroids[k] = pts.mean(axis=0) if len(pts) > 0 else X[rng.randint(N)]
    return labels


# NMI computation
def compute_nmi(labels, true):
    labels = np.array(labels)
    true = np.array(true)
    eps = 1e-10
    N = len(labels)
    clusters = np.unique(labels)
    classes = np.unique(true)
    MI = 0.0
    for c in clusters:
        p_x = (labels == c).sum() / N
        for k in classes:
            p_y = (true == k).sum() / N
            p_xy = ((labels == c) & (true == k)).sum() / N
            if p_xy > 0:
                MI += p_xy * np.log(p_xy / (p_x * p_y) + eps)
    Hx = -sum(
        (labels == c).sum() / N * np.log((labels == c).sum() / N + eps)
        for c in clusters
    )
    Hy = -sum(
        (true == k).sum() / N * np.log((true == k).sum() / N + eps) for k in classes
    )
    return 2 * MI / (Hx + Hy + eps)


class CustomDataset(Dataset):
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


def evaluate(loader, model, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, yb, grp = batch["features"], batch["label"], batch["group"]
            out = model(x)
            losses = criterion(out, yb)
            total_loss += losses.sum().item()
            preds = out.argmax(1)
            for g in (0, 1):
                mask = grp == g
                total[g] += mask.sum().item()
                correct[g] += (preds[mask] == yb[mask]).sum().item()
            total_samples += yb.size(0)
    avg_loss = total_loss / total_samples
    wg_acc = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return avg_loss, wg_acc


class SmallCNN(nn.Module):
    def __init__(self, flat_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flat_dim, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        return self.fc2(x)


# Experiment settings
sp_corr_train = 0.95
sp_corr_eval = 0.5
warmup_epochs = 1
train_epochs = 2
total_epochs = warmup_epochs + train_epochs
batch_size = 64

datasets_list = ["mnist", "fashion_mnist", "cifar10"]
experiment_data = {}

for name in datasets_list:
    raw = load_dataset(name)
    train_raw = (
        raw["train"]
        .filter(lambda ex: ex["label"] in [0, 1])
        .shuffle(seed=0)
        .select(range(2000))
    )
    test_all = (
        raw["test"]
        .filter(lambda ex: ex["label"] in [0, 1])
        .shuffle(seed=0)
        .select(range(1000))
    )
    val_raw = test_all.select(range(500))
    test_raw = test_all.select(range(500, 1000))

    def make_split(ds, corr):
        # pick correct image column
        img_col = "image" if "image" in ds.column_names else "img"
        imgs = ds[img_col]
        ys = ds["label"]
        X, y, g = [], [], []
        for im, lab in zip(imgs, ys):
            arr = np.array(im)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=2)
            grp = lab if random.random() < corr else 1 - lab
            patch = 5
            if grp == 1:
                arr[:patch, :patch, 0] = 255
                arr[:patch, :patch, 1:] = 0
            else:
                arr[:patch, :patch, 2] = 255
                arr[:patch, :patch, :2] = 0
            arr = arr.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            arr = arr.transpose(2, 0, 1)
            X.append(arr)
            y.append(lab)
            g.append(grp)
        return np.stack(X), np.array(y), np.array(g)

    X_train, y_train, g_train = make_split(train_raw, sp_corr_train)
    X_val, y_val, g_val = make_split(val_raw, sp_corr_eval)
    X_test, y_test, g_test = make_split(test_raw, sp_corr_eval)

    train_ds = CustomDataset(X_train, y_train, g_train)
    val_ds = CustomDataset(X_val, y_val, g_val)
    test_ds = CustomDataset(X_test, y_test, g_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    cluster_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    img_size = 28 if name != "cifar10" else 32
    flat_dim = 32 * (img_size // 4) * (img_size // 4)
    model = SmallCNN(flat_dim).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    sample_weights = None
    nmi_val = np.nan
    experiment_data[name] = {
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "train_wg": [],
            "val_wg": [],
            "nmi": [],
        },
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, yb, idxb = batch["features"], batch["label"], batch["idx"]
            out = model(x)
            losses = criterion(out, yb)
            if epoch >= warmup_epochs:
                loss = (losses * sample_weights[idxb]).mean()
            else:
                loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == warmup_epochs - 1:
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
                l = criterion(out, batch["label"]).mean()
                l.backward()
                grads.append(model.fc2.weight.grad.detach().cpu().view(-1).numpy())
            grads = np.stack(grads)
            labels = kmeans_np(grads, 2, 10)
            nmi_val = compute_nmi(labels, g_train)
            counts = np.bincount(labels, minlength=2)
            sw = np.array([1.0 / counts[lab] for lab in labels], dtype=np.float32)
            sample_weights = torch.tensor(sw, device=device)

        tr_loss, tr_wg = evaluate(train_loader, model, criterion)
        v_loss, v_wg = evaluate(val_loader, model, criterion)
        print(f"Epoch {epoch}: validation_loss = {v_loss:.4f}")
        m = experiment_data[name]["metrics"]
        m["train_loss"].append(tr_loss)
        m["val_loss"].append(v_loss)
        m["train_wg"].append(tr_wg)
        m["val_wg"].append(v_wg)
        m["nmi"].append(nmi_val)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            preds.extend(model(batch["features"]).argmax(1).cpu().tolist())
    experiment_data[name]["predictions"] = preds
    experiment_data[name]["ground_truth"] = y_test.tolist()

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
