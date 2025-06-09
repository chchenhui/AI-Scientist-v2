import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms and loaders
normalize = transforms.Normalize((0.1307,), (0.3081,))
train_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_transform = train_transform
aug_transform = transforms.Compose(
    [transforms.RandomRotation(30), transforms.ToTensor(), normalize]
)

train_loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=train_transform),
    batch_size=64,
    shuffle=True,
)
orig_test_loader = DataLoader(
    datasets.MNIST("./data", train=False, download=True, transform=test_transform),
    batch_size=1000,
    shuffle=False,
)
aug_test_loader = DataLoader(
    datasets.MNIST("./data", train=False, download=True, transform=aug_transform),
    batch_size=1000,
    shuffle=False,
)


# Model definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 13 * 13, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# Smoothed CE (epsilon=0 means standard CE)
def smooth_ce(logits, target, epsilon=0.0):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (epsilon / (n - 1)) * (1 - t)
    return -(t * logp).sum(1).mean()


# Training and evaluation loops
def train_one_epoch(model, opt, eps):
    model.train()
    tot = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = smooth_ce(model(x), y, eps)
        loss.backward()
        opt.step()
        tot += loss.item() * x.size(0)
    return tot / len(train_loader.dataset)


def evaluate(model, loader, eps):
    model.eval()
    tot_loss, correct, preds, trues = 0.0, 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            tot_loss += smooth_ce(out, y, eps).item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return tot_loss / len(loader.dataset), correct / len(loader.dataset), preds, trues


# Ablation over weight decay
weight_decays = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
n_epochs = 5
epsilon = 0.0  # no label smoothing here

experiment_data = {"weight_decay": {}}
for wd in weight_decays:
    key = f"wd_{wd}"
    experiment_data["weight_decay"][key] = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, epsilon)
        vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, epsilon)
        _, aug_acc, _, _ = evaluate(model, aug_test_loader, epsilon)
        experiment_data["weight_decay"][key]["losses"]["train"].append(tr_loss)
        experiment_data["weight_decay"][key]["losses"]["val"].append(vl_loss)
        experiment_data["weight_decay"][key]["metrics"]["orig_acc"].append(orig_acc)
        experiment_data["weight_decay"][key]["metrics"]["aug_acc"].append(aug_acc)
        print(
            f"[wd={wd}] Epoch {epoch}/{n_epochs} - tr_loss:{tr_loss:.4f}, "
            f"val_loss:{vl_loss:.4f}, orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
        )
    _, _, pr, gt = evaluate(model, orig_test_loader, epsilon)
    experiment_data["weight_decay"][key]["predictions"] = pr
    experiment_data["weight_decay"][key]["ground_truth"] = gt

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
