import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# create working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
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


# models
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


# loss / utils
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
            p = out.argmax(dim=1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        correct / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(trues),
    )


# hyperparameter grid
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
n_epochs = 5

# container
experiment_data = {"learning_rate": {}}

for lr in learning_rates:
    lr_str = str(lr)
    # init storage
    experiment_data["learning_rate"][lr_str] = {
        "MLP": {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
        },
        "CNN": {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
        },
        "CGR": [],
        "predictions": [],
        "ground_truth": [],
        "predictions_aug": [],
        "ground_truth_aug": [],
    }
    # reinit models & optimizers
    models = {"MLP": MLP().to(device), "CNN": CNN().to(device)}
    optimizers = {name: optim.Adam(m.parameters(), lr=lr) for name, m in models.items()}

    # training loop
    for epoch in range(1, n_epochs + 1):
        # train
        for name in models:
            tl = train_one_epoch(models[name], optimizers[name])
            experiment_data["learning_rate"][lr_str][name]["losses"]["train"].append(tl)
        # eval
        orig_accs, aug_accs = [], []
        for name in models:
            vl, oa, _, _ = evaluate(models[name], orig_test_loader)
            _, aa, _, _ = evaluate(models[name], aug_test_loader)
            experiment_data["learning_rate"][lr_str][name]["losses"]["val"].append(vl)
            experiment_data["learning_rate"][lr_str][name]["metrics"][
                "orig_acc"
            ].append(oa)
            experiment_data["learning_rate"][lr_str][name]["metrics"]["aug_acc"].append(
                aa
            )
            orig_accs.append(oa)
            aug_accs.append(aa)
            print(
                f"LR {lr_str} Model {name} Epoch {epoch}: val_loss={vl:.4f}, orig_acc={oa:.4f}, aug_acc={aa:.4f}"
            )
        # CGR across models
        std_o = np.std(orig_accs)
        std_a = np.std(aug_accs)
        cgr = (std_a - std_o) / (std_o + 1e-8)
        experiment_data["learning_rate"][lr_str]["CGR"].append(cgr)
        print(f"LR {lr_str} Epoch {epoch}: CGR = {cgr:.4f}")

    # final CNN preds
    _, _, p, gt = evaluate(models["CNN"], orig_test_loader)
    experiment_data["learning_rate"][lr_str]["predictions"] = p
    experiment_data["learning_rate"][lr_str]["ground_truth"] = gt
    _, _, pa, ga = evaluate(models["CNN"], aug_test_loader)
    experiment_data["learning_rate"][lr_str]["predictions_aug"] = pa
    experiment_data["learning_rate"][lr_str]["ground_truth_aug"] = ga

# convert lists to arrays
for lr_str, data in experiment_data["learning_rate"].items():
    for m in ["MLP", "CNN"]:
        data[m]["losses"]["train"] = np.array(data[m]["losses"]["train"])
        data[m]["losses"]["val"] = np.array(data[m]["losses"]["val"])
        data[m]["metrics"]["orig_acc"] = np.array(data[m]["metrics"]["orig_acc"])
        data[m]["metrics"]["aug_acc"] = np.array(data[m]["metrics"]["aug_acc"])
    data["CGR"] = np.array(data["CGR"])
    data["predictions"] = np.array(data["predictions"])
    data["ground_truth"] = np.array(data["ground_truth"])
    data["predictions_aug"] = np.array(data["predictions_aug"])
    data["ground_truth_aug"] = np.array(data["ground_truth_aug"])

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
