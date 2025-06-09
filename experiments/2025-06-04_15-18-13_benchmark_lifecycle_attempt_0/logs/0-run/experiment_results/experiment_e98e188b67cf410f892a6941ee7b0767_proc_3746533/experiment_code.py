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

# normalization
normalize = transforms.Normalize((0.1307,), (0.3081,))


# custom Gaussian noise transform
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# augmentation configurations
aug_configs = {
    "no_aug": transforms.Compose([transforms.ToTensor(), normalize]),
    "rot": transforms.Compose(
        [transforms.RandomRotation(30), transforms.ToTensor(), normalize]
    ),
    "rot_trans": transforms.Compose(
        [
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ]
    ),
    "rot_trans_noise": transforms.Compose(
        [
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            AddGaussianNoise(0.0, 0.1),
            normalize,
        ]
    ),
}

# test datasets/loaders
test_transform = transforms.Compose([transforms.ToTensor(), normalize])
rot_test_transform = transforms.Compose(
    [transforms.RandomRotation(30), transforms.ToTensor(), normalize]
)
orig_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)
rot_test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=rot_test_transform
)
orig_test_loader = DataLoader(orig_test_dataset, batch_size=1000, shuffle=False)
rot_test_loader = DataLoader(rot_test_dataset, batch_size=1000, shuffle=False)


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
    total, correct = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = smooth_ce(out, y, epsilon)
            total += loss.item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total / len(loader.dataset), correct / len(loader.dataset), preds, trues


# experiment
epsilon = 0.1
n_epochs = 5
experiment_data = {}

for name, train_transform in aug_configs.items():
    # prepare train loader
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # initialize storage
    experiment_data[name] = {
        "orig": {
            "losses": {"train": [], "val": []},
            "metrics": {"acc": []},
            "predictions": [],
            "ground_truth": [],
        },
        "rot": {
            "losses": {"train": [], "val": []},
            "metrics": {"acc": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
    # model & optimizer
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # training loop
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, epsilon)
        o_val_loss, o_acc, _, _ = evaluate(model, orig_test_loader, epsilon)
        r_val_loss, r_acc, _, _ = evaluate(model, rot_test_loader, epsilon)
        # record
        experiment_data[name]["orig"]["losses"]["train"].append(tr_loss)
        experiment_data[name]["orig"]["losses"]["val"].append(o_val_loss)
        experiment_data[name]["orig"]["metrics"]["acc"].append(o_acc)
        experiment_data[name]["rot"]["losses"]["train"].append(tr_loss)
        experiment_data[name]["rot"]["losses"]["val"].append(r_val_loss)
        experiment_data[name]["rot"]["metrics"]["acc"].append(r_acc)
        print(
            f"[{name}] Epoch {epoch}/{n_epochs} "
            f"tr_loss:{tr_loss:.4f} o_val_loss:{o_val_loss:.4f} o_acc:{o_acc:.4f} r_acc:{r_acc:.4f}"
        )
    # final preds & gts
    _, _, po, go = evaluate(model, orig_test_loader, epsilon)
    _, _, pr, gr = evaluate(model, rot_test_loader, epsilon)
    experiment_data[name]["orig"]["predictions"] = po
    experiment_data[name]["orig"]["ground_truth"] = go
    experiment_data[name]["rot"]["predictions"] = pr
    experiment_data[name]["rot"]["ground_truth"] = gr

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
