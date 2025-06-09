import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# data transforms and loaders
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


# model definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 13 * 13, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# label smoothing loss
def smooth_ce(logits, target, epsilon):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (1 - t) * (epsilon / (n - 1))
    return -(t * logp).sum(dim=1).mean()


# one training epoch
def train_one_epoch(model, optimizer, epsilon):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = smooth_ce(out, y, epsilon)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(train_loader.dataset)


# evaluation on a loader
def evaluate(model, loader, epsilon):
    model.eval()
    total_loss = 0.0
    correct = 0
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


# ablation over learning‐rate schedulers
epsilon = 0.0
n_epochs = 5
experiment_data = {"lr_scheduler": {}}
scheduler_names = ["constant", "step_decay", "cosine_annealing"]

for sched_name in scheduler_names:
    experiment_data["lr_scheduler"][sched_name] = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # choose scheduler
    if sched_name == "step_decay":
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    elif sched_name == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        scheduler = None

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, epsilon)
        if scheduler:
            scheduler.step()
        vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, epsilon)
        _, aug_acc, _, _ = evaluate(model, aug_test_loader, epsilon)
        experiment_data["lr_scheduler"][sched_name]["losses"]["train"].append(tr_loss)
        experiment_data["lr_scheduler"][sched_name]["losses"]["val"].append(vl_loss)
        experiment_data["lr_scheduler"][sched_name]["metrics"]["orig_acc"].append(
            orig_acc
        )
        experiment_data["lr_scheduler"][sched_name]["metrics"]["aug_acc"].append(
            aug_acc
        )
        print(
            f"[{sched_name}] Epoch {epoch}/{n_epochs} - "
            f"tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, "
            f"orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
        )

    # final predictions on original test
    _, _, pr, gt = evaluate(model, orig_test_loader, epsilon)
    experiment_data["lr_scheduler"][sched_name]["predictions"] = pr
    experiment_data["lr_scheduler"][sched_name]["ground_truth"] = gt

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
