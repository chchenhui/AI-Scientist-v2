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


# focal loss
def focal_loss(logits, target, gamma):
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1.0 - p_t) ** gamma) * logp_t
    return loss.mean()


def train_one_epoch_focal(model, optimizer, gamma):
    model.train()
    total = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = focal_loss(out, y, gamma)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(train_loader.dataset)


def evaluate_focal(model, loader, gamma):
    model.eval()
    total_loss = 0.0
    correct = 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = focal_loss(out, y, gamma)
            total_loss += loss.item() * x.size(0)
            p = out.argmax(1)
            correct += p.eq(y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), correct / len(loader.dataset), preds, trues


# ablation study: focal loss
gammas = [0, 1, 2, 5]
n_epochs = 5
experiment_data = {"focal_loss": {}}

for gamma in gammas:
    key = f"gamma_{gamma}"
    experiment_data["focal_loss"][key] = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch_focal(model, optimizer, gamma)
        vl_loss, orig_acc, _, _ = evaluate_focal(model, orig_test_loader, gamma)
        _, aug_acc, _, _ = evaluate_focal(model, aug_test_loader, gamma)
        experiment_data["focal_loss"][key]["losses"]["train"].append(tr_loss)
        experiment_data["focal_loss"][key]["losses"]["val"].append(vl_loss)
        experiment_data["focal_loss"][key]["metrics"]["orig_acc"].append(orig_acc)
        experiment_data["focal_loss"][key]["metrics"]["aug_acc"].append(aug_acc)
        print(
            f"[γ={gamma}] Epoch {epoch}/{n_epochs} - tr_loss:{tr_loss:.4f}, val_loss:{vl_loss:.4f}, orig_acc:{orig_acc:.4f}, aug_acc:{aug_acc:.4f}"
        )
    # final preds & gts on original test
    _, _, pr, gt = evaluate_focal(model, orig_test_loader, gamma)
    experiment_data["focal_loss"][key]["predictions"] = pr
    experiment_data["focal_loss"][key]["ground_truth"] = gt

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
