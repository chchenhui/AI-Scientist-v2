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

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# data transforms
normalize = transforms.Normalize((0.1307,), (0.3081,))
train_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_transform = train_transform
aug_transform = transforms.Compose(
    [transforms.RandomRotation(30), transforms.ToTensor(), normalize]
)

# datasets and loaders
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=train_transform
)
orig_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)
aug_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=aug_transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
orig_test_loader = DataLoader(orig_test_dataset, batch_size=1000, shuffle=False)
aug_test_loader = DataLoader(aug_test_dataset, batch_size=1000, shuffle=False)


# models
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 13 * 13, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class CNN_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.BatchNorm1d(10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# smoothed cross-entropy
def smooth_ce(logits, target, epsilon):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (1 - t) * (epsilon / (n - 1))
    return -(t * logp).sum(dim=1).mean()


# training and evaluation
def train_one_epoch(model, optimizer, eps):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = smooth_ce(out, y, eps)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate(model, loader, eps):
    model.eval()
    total_loss = 0.0
    correct = 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = smooth_ce(out, y, eps)
            total_loss += loss.item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), correct / len(loader.dataset), preds, trues


# experiments
epsilons = [0.0, 0.05, 0.1, 0.2]
n_epochs = 5
variants = [("without_bn", CNN), ("with_bn", CNN_BN)]

experiment_data = {"batch_norm": {}}

for name, ModelClass in variants:
    experiment_data["batch_norm"][name] = {}
    for eps in epsilons:
        key = f"eps_{eps}"
        experiment_data["batch_norm"][name][key] = {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
            "predictions": [],
            "ground_truth": [],
        }
        model = ModelClass().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, n_epochs + 1):
            tr_loss = train_one_epoch(model, optimizer, eps)
            vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, eps)
            _, aug_acc, _, _ = evaluate(model, aug_test_loader, eps)
            d = experiment_data["batch_norm"][name][key]
            d["losses"]["train"].append(tr_loss)
            d["losses"]["val"].append(vl_loss)
            d["metrics"]["orig_acc"].append(orig_acc)
            d["metrics"]["aug_acc"].append(aug_acc)
            print(
                f"[{name} ε={eps}] Epoch {epoch}/{n_epochs} - tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
            )
        # final predictions on original test
        _, _, pr, gt = evaluate(model, orig_test_loader, eps)
        experiment_data["batch_norm"][name][key]["predictions"] = pr
        experiment_data["batch_norm"][name][key]["ground_truth"] = gt

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
