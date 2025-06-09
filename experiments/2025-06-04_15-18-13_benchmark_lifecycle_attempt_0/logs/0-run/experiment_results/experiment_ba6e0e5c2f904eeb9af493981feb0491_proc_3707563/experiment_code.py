import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms and Datasets
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


# Loss & eval functions
criterion = nn.CrossEntropyLoss()


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
    total_loss, correct = 0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        correct / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(trues),
    )


# Hyperparameter tuning for n_epochs
n_epochs_list = [5, 10, 15, 20]
experiment_data = {"n_epochs": {}}

for n_epochs in n_epochs_list:
    print(f"\n=== Running with n_epochs = {n_epochs} ===")
    # Initialize models and optimizers
    models = {"MLP": MLP().to(device), "CNN": CNN().to(device)}
    optimizers = {
        name: optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()
    }
    # Data structure for this run
    run_data = {"models": {}, "cgr": []}
    for name in models:
        run_data["models"][name] = {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
            "predictions": None,
            "ground_truth": None,
            "aug_predictions": None,
            "aug_ground_truth": None,
        }
    # Training loop
    for epoch in range(1, n_epochs + 1):
        # train
        for name in models:
            tl = train_one_epoch(models[name], optimizers[name])
            run_data["models"][name]["losses"]["train"].append(tl)
        # eval and metrics
        orig_accs, aug_accs = [], []
        for name in models:
            vl, oa, _, _ = evaluate(models[name], orig_test_loader)
            run_data["models"][name]["losses"]["val"].append(vl)
            run_data["models"][name]["metrics"]["orig_acc"].append(oa)
            _, aa, _, _ = evaluate(models[name], aug_test_loader)
            run_data["models"][name]["metrics"]["aug_acc"].append(aa)
            orig_accs.append(oa)
            aug_accs.append(aa)
            print(
                f"{name} Epoch {epoch}: val_loss={vl:.4f}, orig_acc={oa:.4f}, aug_acc={aa:.4f}"
            )
        cgr = (np.std(aug_accs) - np.std(orig_accs)) / (np.std(orig_accs) + 1e-8)
        run_data["cgr"].append(cgr)
        print(f"Epoch {epoch}: CGR={cgr:.4f}")
    # Final predictions & ground truth
    for name in models:
        _, _, p, gt = evaluate(models[name], orig_test_loader)
        run_data["models"][name]["predictions"] = p
        run_data["models"][name]["ground_truth"] = gt
        _, _, pa, gta = evaluate(models[name], aug_test_loader)
        run_data["models"][name]["aug_predictions"] = pa
        run_data["models"][name]["aug_ground_truth"] = gta
    # Store run data
    experiment_data["n_epochs"][str(n_epochs)] = run_data

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
