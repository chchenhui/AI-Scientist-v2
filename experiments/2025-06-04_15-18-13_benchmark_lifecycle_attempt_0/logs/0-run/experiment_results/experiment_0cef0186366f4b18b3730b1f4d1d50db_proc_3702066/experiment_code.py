# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
normalize = transforms.Normalize((0.1307,), (0.3081,))
train_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_transform = train_transform
aug_transform = transforms.Compose(
    [transforms.RandomRotation(30), transforms.ToTensor(), normalize]
)

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


# Model definitions
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 13 * 13, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


models = {"MLP": MLP().to(device), "CNN": CNN().to(device)}
optimizers = {name: optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()}
criterion = nn.CrossEntropyLoss()

# Experiment data
experiment_data = {
    "original": {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    },
    "CGR": [],
}


def train_one_epoch(model, optimizer):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            all_preds.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        correct / len(loader.dataset),
        np.concatenate(all_preds),
        np.concatenate(all_true),
    )


# Training loop
n_epochs = 5
for epoch in range(1, n_epochs + 1):
    # Train both models
    for name in models:
        train_loss = train_one_epoch(models[name], optimizers[name])
        experiment_data["original"]["losses"]["train"].append(train_loss)
    # Evaluate
    orig_accs = []
    aug_accs = []
    for name in models:
        val_loss, orig_acc, _, _ = evaluate(models[name], orig_test_loader)
        experiment_data["original"]["losses"]["val"].append(val_loss)
        print(f"Model {name} Epoch {epoch}: validation_loss = {val_loss:.4f}")
        _, aug_acc, _, _ = evaluate(models[name], aug_test_loader)
        experiment_data["original"]["metrics"]["orig_acc"].append(orig_acc)
        experiment_data["original"]["metrics"]["aug_acc"].append(aug_acc)
        orig_accs.append(orig_acc)
        aug_accs.append(aug_acc)
    # Compute CGR
    std_orig = np.std(orig_accs)
    std_aug = np.std(aug_accs)
    cgr = (std_aug - std_orig) / (std_orig + 1e-8)
    experiment_data["CGR"].append(cgr)
    print(f"Epoch {epoch}: CGR = {cgr:.4f}")

# Save final predictions and ground truth for both splits (using CNN)
_, _, preds, gts = evaluate(models["CNN"], orig_test_loader)
experiment_data["original"]["predictions"] = preds
experiment_data["original"]["ground_truth"] = gts
_, _, preds_aug, gts_aug = evaluate(models["CNN"], aug_test_loader)
experiment_data["augmented"] = {"predictions": preds_aug, "ground_truth": gts_aug}

# Save all metrics and data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
