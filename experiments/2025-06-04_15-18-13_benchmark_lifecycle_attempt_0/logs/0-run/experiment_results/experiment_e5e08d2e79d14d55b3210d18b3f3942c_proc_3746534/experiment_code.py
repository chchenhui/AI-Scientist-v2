import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data transforms
normalize = transforms.Normalize((0.1307,), (0.3081,))
train_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_transform = train_transform
aug_transform = transforms.Compose(
    [transforms.RandomRotation(30), transforms.ToTensor(), normalize]
)

# datasets & loaders
train_ds = datasets.MNIST(
    root="./data", train=True, download=True, transform=train_transform
)
orig_test_ds = datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)
aug_test_ds = datasets.MNIST(
    root="./data", train=False, download=True, transform=aug_transform
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
orig_test_loader = DataLoader(orig_test_ds, batch_size=1000, shuffle=False)
aug_test_loader = DataLoader(aug_test_ds, batch_size=1000, shuffle=False)


# simple CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 13 * 13, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# mixup utility
def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        x2, y2 = x[idx], y[idx]
        return lam * x + (1 - lam) * x2, y, y2, lam
    else:
        return x, y, y, 1.0


# one training epoch with mixup
def train_one_epoch(model, opt, alpha):
    model.train()
    tot, cnt = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha)
        out = model(x_mix)
        if alpha > 0:
            loss = lam * F.cross_entropy(out, y_a) + (1 - lam) * F.cross_entropy(
                out, y_b
            )
        else:
            loss = F.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * x.size(0)
        cnt += x.size(0)
    return tot / cnt


# evaluation
def evaluate(model, loader):
    model.eval()
    tot, corr, cnt = 0.0, 0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            tot += loss.item() * x.size(0)
            p = out.argmax(1)
            corr += p.eq(y).sum().item()
            cnt += x.size(0)
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    return tot / cnt, corr / cnt, np.concatenate(preds), np.concatenate(trues)


# ablation over mixup strengths
alphas = [0.0, 0.2, 0.4, 0.8]
n_epochs = 5
experiment_data = {"mixup": {}}

for alpha in alphas:
    key = f"alpha_{alpha}"
    experiment_data["mixup"][key] = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": None,
        "ground_truth": None,
    }
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, alpha)
        vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader)
        _, aug_acc, _, _ = evaluate(model, aug_test_loader)
        experiment_data["mixup"][key]["losses"]["train"].append(tr_loss)
        experiment_data["mixup"][key]["losses"]["val"].append(vl_loss)
        experiment_data["mixup"][key]["metrics"]["orig_acc"].append(orig_acc)
        experiment_data["mixup"][key]["metrics"]["aug_acc"].append(aug_acc)
        print(
            f"[α={alpha}] Epoch {epoch}/{n_epochs} - tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
        )
    # final preds/gts
    _, _, pr, gt = evaluate(model, orig_test_loader)
    experiment_data["mixup"][key]["predictions"] = pr
    experiment_data["mixup"][key]["ground_truth"] = gt

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
