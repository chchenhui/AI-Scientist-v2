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


# smoothed cross‐entropy
def smooth_ce(logits, target, epsilon):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (1 - t) * (epsilon / (n - 1))
    loss = -(t * logp).sum(dim=1)
    return loss.mean()


def train_one_epoch(model, optimizer, epsilon):
    model.train()
    total = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = smooth_ce(out, y, epsilon)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(train_loader.dataset)


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


# CNN with variable depth and roughly constant param budget
class CNN(nn.Module):
    def __init__(self, depth):
        super().__init__()
        # choose widths so total params ~174k for depth=1,2,3
        width_map = {1: 16, 2: 76, 3: 96}
        out_ch = width_map[depth]
        layers = []
        in_ch = 1
        for _ in range(depth):
            layers += [nn.Conv2d(in_ch, out_ch, 3), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        # compute spatial size
        h = 28
        for _ in range(depth):
            h = (h - 2) // 2
        flat_dim = h * h * out_ch
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(flat_dim, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# run ablation: vary depth and label smoothing
epsilons = [0.0, 0.05, 0.1, 0.2]
depths = [1, 2, 3]
n_epochs = 5

experiment_data = {"network_depth": {}}

for d in depths:
    key_d = f"depth_{d}"
    experiment_data["network_depth"][key_d] = {}
    for eps in epsilons:
        key_e = f"eps_{eps}"
        experiment_data["network_depth"][key_d][key_e] = {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
            "predictions": [],
            "ground_truth": [],
        }
        model = CNN(depth=d).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, n_epochs + 1):
            tr_loss = train_one_epoch(model, optimizer, eps)
            vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, eps)
            _, aug_acc, _, _ = evaluate(model, aug_test_loader, eps)
            experiment_data["network_depth"][key_d][key_e]["losses"]["train"].append(
                tr_loss
            )
            experiment_data["network_depth"][key_d][key_e]["losses"]["val"].append(
                vl_loss
            )
            experiment_data["network_depth"][key_d][key_e]["metrics"][
                "orig_acc"
            ].append(orig_acc)
            experiment_data["network_depth"][key_d][key_e]["metrics"]["aug_acc"].append(
                aug_acc
            )
            print(
                f"[depth={d} ε={eps}] Epoch {epoch}/{n_epochs} - "
                f"tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, "
                f"orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
            )
        # final preds & gts on original test
        _, _, pr, gt = evaluate(model, orig_test_loader, eps)
        experiment_data["network_depth"][key_d][key_e]["predictions"] = pr
        experiment_data["network_depth"][key_d][key_e]["ground_truth"] = gt

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
