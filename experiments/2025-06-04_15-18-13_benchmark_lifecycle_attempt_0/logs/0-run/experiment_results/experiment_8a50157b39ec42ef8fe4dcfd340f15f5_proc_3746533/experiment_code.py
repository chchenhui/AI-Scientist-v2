import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

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


# smoothed cross-entropy
def smooth_ce(logits, target, epsilon):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (1 - t) * (epsilon / (n - 1))
    return -(t * logp).sum(dim=1).mean()


# training and evaluation
def train_one_epoch(model, optimizer, epsilon):
    model.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = smooth_ce(out, y, epsilon)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(train_loader.dataset)


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


# weight initialization factory
def get_init_fn(scheme):
    def init_fn(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return init_fn


# ablation over weight initialization schemes
schemes = ["xavier_uniform", "kaiming_normal", "orthogonal"]
n_epochs = 5
epsilon = 0.1
lr = 1e-3

experiment_data = {"weight_initialization": {}}

for scheme in schemes:
    experiment_data["weight_initialization"][scheme] = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = CNN().to(device)
    model.apply(get_init_fn(scheme))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, epsilon)
        vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, epsilon)
        _, aug_acc, _, _ = evaluate(model, aug_test_loader, epsilon)
        experiment_data["weight_initialization"][scheme]["losses"]["train"].append(
            tr_loss
        )
        experiment_data["weight_initialization"][scheme]["losses"]["val"].append(
            vl_loss
        )
        experiment_data["weight_initialization"][scheme]["metrics"]["orig_acc"].append(
            orig_acc
        )
        experiment_data["weight_initialization"][scheme]["metrics"]["aug_acc"].append(
            aug_acc
        )
        print(
            f"[{scheme}] Epoch {epoch}/{n_epochs} - tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
        )
    # final preds & gt on original test
    _, _, pr, gt = evaluate(model, orig_test_loader, epsilon)
    experiment_data["weight_initialization"][scheme]["predictions"] = pr
    experiment_data["weight_initialization"][scheme]["ground_truth"] = gt

# save all collected data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
