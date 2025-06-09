import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from datasets import load_dataset
import math

# setup working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# simple k-means + centroids
def kmeans_np(X, n_clusters, n_iters=10):
    rng = np.random.RandomState(0)
    N, D = X.shape
    idx = rng.choice(N, n_clusters, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(n_iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new = dists.argmin(axis=1)
        if np.all(new == labels):
            break
        labels = new
        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                centroids[k] = pts.mean(0)
            else:
                centroids[k] = X[rng.randint(N)]
    return labels, centroids


# compute NMI
def compute_nmi(labels, groups):
    N = len(labels)
    cnt_u = {}
    cnt_v = {}
    for u in labels:
        cnt_u[u] = cnt_u.get(u, 0) + 1
    for v in groups:
        cnt_v[v] = cnt_v.get(v, 0) + 1
    p_u = {u: cnt_u[u] / N for u in cnt_u}
    p_v = {v: cnt_v[v] / N for v in cnt_v}
    joint = {}
    for u, v in zip(labels, groups):
        joint[(u, v)] = joint.get((u, v), 0) + 1
    p_uv = {k: joint[k] / N for k in joint}
    H_u = -sum(p * math.log(p + 1e-8) for p in p_u.values())
    H_v = -sum(p * math.log(p + 1e-8) for p in p_v.values())
    I = 0.0
    for (u, v), p in p_uv.items():
        I += p * (
            math.log(p + 1e-8) - math.log(p_u[u] + 1e-8) - math.log(p_v[v] + 1e-8)
        )
    return I / (((H_u + H_v) / 2) + 1e-8)


# evaluation: worst-group accuracy & loss sum
def evaluate(loader, model, criterion):
    model.eval()
    loss_sum = 0.0
    correct = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y, g = batch["features"], batch["label"], batch["group"]
            out = model(x)
            losses = criterion(out, y)
            loss_sum += losses.sum().item()
            preds = out.argmax(1)
            for gg in (0, 1):
                m = g == gg
                total[gg] += m.sum().item()
                if m.sum().item() > 0:
                    correct[gg] += (preds[m] == y[m]).sum().item()
    wg = min(correct[g] / total[g] if total[g] > 0 else 0.0 for g in (0, 1))
    return loss_sum, wg


# small CNN
class SmallCNN(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        h = self.hidden(x).view(x.size(0), -1)
        return self.classifier(h)


# colored spurious dataset wrapper
class ColoredHFImageDataset(Dataset):
    def __init__(self, hf_ds, sp_corr, mean, std):
        self.ds = hf_ds
        self.sp = sp_corr
        self.norm = T.Normalize(mean, std)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        idx = int(idx)
        item = self.ds[idx]
        img = item.get("image", item.get("img"))
        label = item["label"]
        flip = np.random.rand() < self.sp
        sp = (label % 2) if flip else 1 - (label % 2)
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, 2)
        col = np.zeros_like(arr)
        if sp == 0:
            col[..., 0] = arr[..., 0]
        else:
            col[..., 1] = arr[..., 0]
        img2 = Image.fromarray(col)
        x = T.ToTensor()(img2)
        x = self.norm(x)
        return {
            "features": x,
            "label": torch.tensor(label, dtype=torch.long),
            "group": torch.tensor(sp, dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


# experiment across three HF datasets
datasets_cfg = {
    "mnist": ([0.5] * 3, [0.5] * 3),
    "fashion_mnist": ([0.5] * 3, [0.5] * 3),
    "cifar10": ([0.5] * 3, [0.5] * 3),
}
spurious_corr = 0.9
results = {}

for name, (mean, std) in datasets_cfg.items():
    print(f"\n*** DATASET {name} ***")
    ds = load_dataset(name)
    split = ds["train"].train_test_split(0.2, seed=0)
    train_hf, val_hf = split["train"], split["test"]
    test_hf = ds["test"]
    train_ds = ColoredHFImageDataset(train_hf, spurious_corr, mean, std)
    val_ds = ColoredHFImageDataset(val_hf, spurious_corr, mean, std)
    test_ds = ColoredHFImageDataset(test_hf, spurious_corr, mean, std)
    N = len(train_ds)
    bs = 128
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SmallCNN(
        3,
        (
            len(ds["train"].features["label"].names)
            if hasattr(ds["train"], "features")
            else len(set(ds["train"]["label"]))
        ),
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    warmup, train_ep = 1, 2
    total = warmup + train_ep
    sw = None
    val_wg, nmi_list = [], []
    for epoch in range(total):
        model.train()
        for batch in train_loader:
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            x, y, idx = batch["features"], batch["label"], batch["idx"]
            out = model(x)
            losses = criterion(out, y)
            if epoch > warmup and sw is not None:
                loss = (losses * sw[idx]).mean()
            else:
                loss = losses.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == warmup:
            sub = np.random.choice(N, min(N, 2000), replace=False).tolist()
            subset = Subset(train_ds, sub)
            loader = DataLoader(subset, batch_size=1, shuffle=False)
            grads, true_g = [], []
            for b in loader:
                b = {
                    k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)
                }
                optimizer.zero_grad()
                out = model.hidden(b["features"]).view(1, -1)
                logits = model.classifier(out)
                l = criterion(logits, b["label"])
                l.mean().backward()
                g1 = model.classifier.weight.grad.detach().cpu().view(-1).numpy()
                grads.append(g1)
                true_g.append(int(b["group"].cpu().item()))
            grads = np.stack(grads)
            best = (-1, None, None)
            for k in (2, 3):
                lbls, cents = kmeans_np(grads, k)
                d_intra = ((grads - cents[lbls]) ** 2).sum(1).mean()
                d_inter = (
                    np.mean(
                        [
                            ((cents[i] - cents[j]) ** 2).sum()
                            for i in range(k)
                            for j in range(i + 1, k)
                        ]
                    )
                    if k > 1
                    else 0
                )
                score = d_inter / (d_intra + 1e-8)
                if score > best[0]:
                    best = (score, lbls, cents)
            _, sub_lbls, cents = best
            cnt = np.bincount(sub_lbls, minlength=cents.shape[0])
            sw_arr = np.ones(N, dtype=np.float32)
            for i, lab in enumerate(sub_lbls):
                sw_arr[sub[i]] = 1.0 / (cnt[lab] + 1e-8)
            sw = torch.tensor(sw_arr, device=device)
            nmi0 = compute_nmi(sub_lbls, true_g)

        lval, wg = evaluate(val_loader, model, criterion)
        val_wg.append(wg)
        nmi_list.append(nmi0 if epoch >= warmup else 0.0)
        print(
            f"[{name}] Epoch {epoch}: validation_loss = {lval/len(val_ds):.4f}, val_wg = {wg:.4f}, NMI = {nmi_list[-1]:.4f}"
        )

    results[name] = {"metrics": {"val_wg": val_wg}, "nmi": nmi_list}

# save data
experiment_data = results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Saved experiment data to {working_dir}/experiment_data.npy")
