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
print(f"Using device: {device}")

# data
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


# model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 13 * 13, 64), nn.ReLU(), nn.Linear(64, 10)
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


def train_one_epoch(model, optimizer, epsilon):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = smooth_ce(out, y, epsilon)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate(model, loader, epsilon):
    model.eval()
    total_loss, correct = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = smooth_ce(out, y, epsilon)
            total_loss += loss.item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), correct / len(loader.dataset), preds, trues


# ablation over optimizer choice
optimizers = {
    "Adam": lambda params: optim.Adam(params, lr=1e-3),
    "SGD": lambda params: optim.SGD(params, lr=1e-3, momentum=0.9),
    "RMSprop": lambda params: optim.RMSprop(params, lr=1e-3),
    "Adagrad": lambda params: optim.Adagrad(params, lr=1e-3),
}

epsilon = 0.1
n_epochs = 5
experiment_data = {"optimizer_choice": {}}

for opt_name, opt_fn in optimizers.items():
    experiment_data["optimizer_choice"][opt_name] = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = CNN().to(device)
    optimizer = opt_fn(model.parameters())
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, epsilon)
        val_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, epsilon)
        _, aug_acc, _, _ = evaluate(model, aug_test_loader, epsilon)
        experiment_data["optimizer_choice"][opt_name]["losses"]["train"].append(tr_loss)
        experiment_data["optimizer_choice"][opt_name]["losses"]["val"].append(val_loss)
        experiment_data["optimizer_choice"][opt_name]["metrics"]["orig_acc"].append(
            orig_acc
        )
        experiment_data["optimizer_choice"][opt_name]["metrics"]["aug_acc"].append(
            aug_acc
        )
        print(
            f"[Optimizer={opt_name}] Epoch {epoch}/{n_epochs} - tr_loss:{tr_loss:.4f}, val_loss:{val_loss:.4f}, orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
        )
    # final predictions & ground truth
    _, _, preds, gt = evaluate(model, orig_test_loader, epsilon)
    experiment_data["optimizer_choice"][opt_name]["predictions"] = preds
    experiment_data["optimizer_choice"][opt_name]["ground_truth"] = gt

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
