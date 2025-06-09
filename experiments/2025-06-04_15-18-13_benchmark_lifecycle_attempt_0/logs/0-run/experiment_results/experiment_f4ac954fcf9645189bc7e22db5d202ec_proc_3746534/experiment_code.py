import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# transforms
normalize = transforms.Normalize((0.1307,), (0.3081,))
train_transform = transforms.Compose([transforms.ToTensor(), normalize])
test_transform = train_transform
aug_transform = transforms.Compose(
    [transforms.RandomRotation(30), transforms.ToTensor(), normalize]
)

# test loaders
orig_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)
aug_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=aug_transform
)
orig_test_loader = DataLoader(orig_test, batch_size=1000, shuffle=False)
aug_test_loader = DataLoader(aug_test, batch_size=1000, shuffle=False)


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


# loss with label smoothing
def smooth_ce(logits, target, epsilon):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (1 - t) * (epsilon / (n - 1))
    return -(t * logp).sum(dim=1).mean()


# training and evaluation
def train_one_epoch(model, optimizer, eps, loader):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = smooth_ce(model(x), y, eps)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, eps):
    model.eval()
    total, correct = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total += smooth_ce(out, y, eps).item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total / len(loader.dataset), correct / len(loader.dataset), preds, trues


# load clean train targets for reuse
base_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=train_transform
)
orig_targets = base_train.targets.clone()

# ablation hyperparams
noise_levels = [0.0, 0.1, 0.2, 0.3]
epsilons = [0.0, 0.05, 0.1, 0.2]
n_epochs = 5
batch_size = 64

experiment_data = {"label_noise": {}}

for noise in noise_levels:
    # build noisy dataset
    ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    t = orig_targets.clone()
    if noise > 0:
        num = int(noise * len(t))
        idxs = np.random.choice(len(t), num, replace=False)
        for i in idxs:
            orig = t[i].item()
            nl = torch.randint(0, 9, (1,)).item()
            if nl >= orig:
                nl += 1
            t[i] = nl
    ds.targets = t
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    noise_key = f"noise_{noise}"
    experiment_data["label_noise"][noise_key] = {}
    for eps in epsilons:
        key = f"smooth_{eps}"
        experiment_data["label_noise"][noise_key][key] = {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
            "predictions": [],
            "ground_truth": [],
        }
        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, n_epochs + 1):
            tr_loss = train_one_epoch(model, optimizer, eps, train_loader)
            vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, eps)
            _, aug_acc, _, _ = evaluate(model, aug_test_loader, eps)
            ed = experiment_data["label_noise"][noise_key][key]
            ed["losses"]["train"].append(tr_loss)
            ed["losses"]["val"].append(vl_loss)
            ed["metrics"]["orig_acc"].append(orig_acc)
            ed["metrics"]["aug_acc"].append(aug_acc)
            print(
                f"[noise={noise}, ε={eps}] Epoch {epoch}/{n_epochs} "
                f"tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, "
                f"orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
            )
        _, _, pr, gt = evaluate(model, orig_test_loader, eps)
        ed["predictions"] = pr
        ed["ground_truth"] = gt

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
