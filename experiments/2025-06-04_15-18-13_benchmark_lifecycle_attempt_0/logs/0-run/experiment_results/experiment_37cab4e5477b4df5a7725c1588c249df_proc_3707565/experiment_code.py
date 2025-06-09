import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# prepare working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# transforms and datasets
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

orig_test_loader = DataLoader(orig_test_dataset, batch_size=1000, shuffle=False)
aug_test_loader = DataLoader(aug_test_dataset, batch_size=1000, shuffle=False)


# model definitions
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


# utility functions
criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, opt, loader):
    model.train()
    tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        l = criterion(model(x), y)
        l.backward()
        opt.step()
        tot += l.item() * x.size(0)
    return tot / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    tot = 0
    corr = 0
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            tot += criterion(out, y).item() * x.size(0)
            p = out.argmax(1)
            corr += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            gts.append(y.cpu().numpy())
    return (
        tot / len(loader.dataset),
        corr / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(gts),
    )


# hyperparameter sweep
batch_sizes = [32, 64, 128, 256]
n_epochs = 5
experiment_data = {"batch_size_sweep": {}}

for bs in batch_sizes:
    key = str(bs)
    experiment_data["batch_size_sweep"][key] = {}
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    for model_name, ModelClass in [("MLP", MLP), ("CNN", CNN)]:
        # initialize
        model = ModelClass().to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        # containers
        mdata = {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
            "predictions": [],
            "ground_truth": [],
        }
        # train
        for epoch in range(1, n_epochs + 1):
            trl = train_one_epoch(model, opt, train_loader)
            mdata["losses"]["train"].append(trl)
            vl, oa, _, _ = evaluate(model, orig_test_loader)
            _, aa, _, _ = evaluate(model, aug_test_loader)
            mdata["losses"]["val"].append(vl)
            mdata["metrics"]["orig_acc"].append(oa)
            mdata["metrics"]["aug_acc"].append(aa)
            print(
                f"BS {bs} {model_name} E{epoch}: tr_loss {trl:.4f}, val_loss {vl:.4f}, orig_acc {oa:.4f}, aug_acc {aa:.4f}"
            )
        # final preds
        _, _, preds, gts = evaluate(model, orig_test_loader)
        mdata["predictions"] = preds
        mdata["ground_truth"] = gts
        # store
        experiment_data["batch_size_sweep"][key][model_name] = mdata

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
