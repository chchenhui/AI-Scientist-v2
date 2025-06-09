import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Setup working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
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


# Models
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


# Criterion
criterion = nn.CrossEntropyLoss()


# Helpers
def train_one_epoch(model, optimizer):
    model.train()
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(train_loader.dataset)


def evaluate(model, loader):
    model.eval()
    total, corr = 0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total += loss.item() * x.size(0)
            p = out.argmax(dim=1)
            corr += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    return (
        total / len(loader.dataset),
        corr / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(trues),
    )


# Hyperparam tuning over beta1
betas1 = [0.8, 0.85, 0.95]
n_epochs = 5
experiment_data = {"adam_beta1": {}}

for b1 in betas1:
    key = f"beta1_{b1}"
    experiment_data["adam_beta1"][key] = {
        "MLP": {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
            "predictions": [],
            "ground_truth": [],
        },
        "CNN": {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "aug_acc": []},
            "predictions": [],
            "ground_truth": [],
            "predictions_aug": [],
            "ground_truth_aug": [],
        },
        "CGR": [],
    }
    # init models & optimizers
    models = {"MLP": MLP().to(device), "CNN": CNN().to(device)}
    opts = {
        n: optim.Adam(m.parameters(), lr=1e-3, betas=(b1, 0.999))
        for n, m in models.items()
    }
    # train epochs
    for epoch in range(n_epochs):
        orig_accs, aug_accs = [], []
        for name, m in models.items():
            tr_loss = train_one_epoch(m, opts[name])
            experiment_data["adam_beta1"][key][name]["losses"]["train"].append(tr_loss)
        for name, m in models.items():
            v_loss, o_acc, _, _ = evaluate(m, orig_test_loader)
            _, a_acc, _, _ = evaluate(m, aug_test_loader)
            experiment_data["adam_beta1"][key][name]["losses"]["val"].append(v_loss)
            experiment_data["adam_beta1"][key][name]["metrics"]["orig_acc"].append(
                o_acc
            )
            experiment_data["adam_beta1"][key][name]["metrics"]["aug_acc"].append(a_acc)
            orig_accs.append(o_acc)
            aug_accs.append(a_acc)
        std_o = np.std(orig_accs)
        std_a = np.std(aug_accs)
        cgr = (std_a - std_o) / (std_o + 1e-8)
        experiment_data["adam_beta1"][key]["CGR"].append(cgr)
        print(f"beta1={b1} Epoch {epoch+1}/{n_epochs} CGR={cgr:.4f}")
    # save final CNN predictions
    _, _, p_o, g_o = evaluate(models["CNN"], orig_test_loader)
    _, _, p_a, g_a = evaluate(models["CNN"], aug_test_loader)
    cd = experiment_data["adam_beta1"][key]["CNN"]
    cd["predictions"] = p_o
    cd["ground_truth"] = g_o
    cd["predictions_aug"] = p_a
    cd["ground_truth_aug"] = g_a

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
