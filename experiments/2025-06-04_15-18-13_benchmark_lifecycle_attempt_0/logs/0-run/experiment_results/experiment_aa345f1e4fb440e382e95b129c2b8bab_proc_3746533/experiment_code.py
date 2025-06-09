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


# smoothed CE
def smooth_ce(logits, target, epsilon):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - epsilon) + (1 - t) * (epsilon / (n - 1))
    return -(t * logp).sum(dim=1).mean()


# train and eval
def train_one_epoch(model, opt, eps):
    model.train()
    total = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = smooth_ce(model(x), y, eps)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(train_loader.dataset)


def evaluate(model, loader, eps):
    model.eval()
    total_loss, correct = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += smooth_ce(out, y, eps).item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), correct / len(loader.dataset), preds, trues


# CNN parameterized by activation
class CNN(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1), act(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 13 * 13, 64), act(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ablation settings
activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "SELU": nn.SELU,
    "GELU": nn.GELU,
}

experiment_data = {"activation_function_ablation": {}}
epsilon = 0.1
n_epochs = 5

for name, act in activations.items():
    data = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = CNN(act).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, opt, epsilon)
        vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, epsilon)
        _, aug_acc, _, _ = evaluate(model, aug_test_loader, epsilon)
        data["losses"]["train"].append(tr_loss)
        data["losses"]["val"].append(vl_loss)
        data["metrics"]["orig_acc"].append(orig_acc)
        data["metrics"]["aug_acc"].append(aug_acc)
        print(
            f"[{name}] Epoch {epoch}/{n_epochs} - tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
        )
    # final predictions on original test
    _, _, pr, gt = evaluate(model, orig_test_loader, epsilon)
    data["predictions"] = pr
    data["ground_truth"] = gt
    experiment_data["activation_function_ablation"][name] = data

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
