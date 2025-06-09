import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# setup working directory and reproducibility
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# transforms
normalize = transforms.Normalize((0.1307,), (0.3081,))
train_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_transform = train_transform

# datasets to use
dataset_info = {
    "MNIST": {"class": datasets.MNIST, "kwargs": {}},
    "FashionMNIST": {"class": datasets.FashionMNIST, "kwargs": {}},
    "KMNIST": {"class": datasets.KMNIST, "kwargs": {}},
}


# simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# smoothed cross‐entropy
def smooth_ce(logits, target, epsilon):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (1 - t) * (epsilon / (n - 1))
    return -(t * logp).sum(dim=1).mean()


# one‐epoch train
def train_one_epoch(model, optimizer, loader, epsilon):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = smooth_ce(out, y, epsilon)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# evaluation with standard CE loss
def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)  # use standard CE here
            total_loss += loss.item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    n = len(loader.dataset)
    return total_loss / n, correct / n, preds, trues


# run experiments
epsilons = [0.0, 0.1]
n_epochs = 5
experiment_data = {"multi_dataset_generalization": {}}

for ds_name, info in dataset_info.items():
    experiment_data["multi_dataset_generalization"][ds_name] = {}
    cls = info["class"]
    kwargs = info["kwargs"]
    train_ds = cls(
        root="./data", train=True, download=True, transform=train_transform, **kwargs
    )
    test_ds = cls(
        root="./data", train=False, download=True, transform=test_transform, **kwargs
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    for eps in epsilons:
        key = f"eps_{eps}"
        ed = {
            "losses": {"train": [], "val": []},
            "metrics": {"train": [], "val": []},
            "predictions": None,
            "ground_truth": None,
        }
        experiment_data["multi_dataset_generalization"][ds_name][key] = ed

        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, optimizer, train_loader, eps)
            val_loss, val_acc, _, _ = evaluate(model, test_loader)

            ed["losses"]["train"].append(tr_loss)
            ed["losses"]["val"].append(val_loss)
            ed["metrics"]["train"].append(tr_acc)
            ed["metrics"]["val"].append(val_acc)

            print(f"[{ds_name} ε={eps}] Epoch {epoch}/{n_epochs}")
            print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

        _, _, pr, gt = evaluate(model, test_loader)
        ed["predictions"] = pr
        ed["ground_truth"] = gt

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
