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


# smoothed cross-entropy
def smooth_ce(logits, target, eps):
    logp = F.log_softmax(logits, dim=1)
    n = logits.size(1)
    with torch.no_grad():
        t = torch.zeros_like(logp).scatter_(1, target.unsqueeze(1), 1)
        t = t * (1 - eps) + (1 - t) * (eps / (n - 1))
    return -(t * logp).sum(dim=1).mean()


# standard training loop
def train_one_epoch(model, optimizer, eps_ls):
    model.train()
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = smooth_ce(out, y, eps_ls)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(train_loader.dataset)


# standard evaluation
def evaluate(model, loader, eps_ls):
    model.eval()
    total, correct = 0, 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = smooth_ce(out, y, eps_ls)
            total += loss.item() * x.size(0)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    return (
        total / len(loader.dataset),
        correct / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(trues),
    )


# FGSM attack helper
mean = torch.tensor((0.1307,), device=device).view(1, 1, 1)
std = torch.tensor((0.3081,), device=device).view(1, 1, 1)
x_min = (0 - mean) / std
x_max = (1 - mean) / std


def fgsm_attack(model, x, y, epsilon_adv, eps_ls):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = smooth_ce(logits, y, eps_ls)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + epsilon_adv * grad.sign()
    return torch.clamp(x_adv, x_min, x_max).detach()


# adversarial training loop
def train_one_epoch_adv(model, optimizer, eps_ls, epsilon_adv):
    model.train()
    total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_clean = smooth_ce(out, y, eps_ls)
        x_adv = fgsm_attack(model, x, y, epsilon_adv, eps_ls)
        out_adv = model(x_adv)
        loss_adv = smooth_ce(out_adv, y, eps_ls)
        loss = 0.5 * (loss_clean + loss_adv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(train_loader.dataset)


# adversarial evaluation
def evaluate_adv(model, loader, epsilon_adv, eps_ls):
    model.eval()
    total, correct = 0, 0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = fgsm_attack(model, x, y, epsilon_adv, eps_ls)
        out = model(x_adv)
        total += smooth_ce(out, y, eps_ls).item() * x.size(0)
        p = out.argmax(1)
        correct += (p == y).sum().item()
        preds.append(p.cpu().numpy())
        trues.append(y.cpu().numpy())
    return (
        total / len(loader.dataset),
        correct / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(trues),
    )


# hyperparameters
epsilons = [0.0, 0.05, 0.1, 0.2]
epsilon_adv = 0.1
n_epochs = 5

# experiment data
experiment_data = {"label_smoothing": {}, "adversarial_training": {}}

# baseline label smoothing experiments
for eps in epsilons:
    key = f"eps_{eps}"
    experiment_data["label_smoothing"][key] = {
        "losses": {"train": [], "val": []},
        "metrics": {"orig_acc": [], "aug_acc": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, eps)
        vl_loss, orig_acc, _, _ = evaluate(model, orig_test_loader, eps)
        _, aug_acc, _, _ = evaluate(model, aug_test_loader, eps)
        experiment_data["label_smoothing"][key]["losses"]["train"].append(tr_loss)
        experiment_data["label_smoothing"][key]["losses"]["val"].append(vl_loss)
        experiment_data["label_smoothing"][key]["metrics"]["orig_acc"].append(orig_acc)
        experiment_data["label_smoothing"][key]["metrics"]["aug_acc"].append(aug_acc)
        print(
            f"[LS eps={eps}] Epoch {epoch}/{n_epochs} tr_loss:{tr_loss:.4f} val_loss:{vl_loss:.4f} orig_acc:{orig_acc:.4f} aug_acc:{aug_acc:.4f}"
        )
    _, _, pr, gt = evaluate(model, orig_test_loader, eps)
    experiment_data["label_smoothing"][key]["predictions"] = pr
    experiment_data["label_smoothing"][key]["ground_truth"] = gt

# adversarial training ablation
for eps in epsilons:
    key = f"eps_{eps}"
    experiment_data["adversarial_training"][key] = {
        "clean": {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "robust_acc": []},
            "predictions": [],
            "ground_truth": [],
        },
        "adv": {
            "losses": {"train": [], "val": []},
            "metrics": {"orig_acc": [], "robust_acc": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
    # clean training baseline
    model_C = CNN().to(device)
    opt_C = optim.Adam(model_C.parameters(), lr=1e-3)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model_C, opt_C, eps)
        vl_loss, orig_acc, _, _ = evaluate(model_C, orig_test_loader, eps)
        _, robust_acc, _, _ = evaluate_adv(model_C, orig_test_loader, epsilon_adv, eps)
        experiment_data["adversarial_training"][key]["clean"]["losses"]["train"].append(
            tr_loss
        )
        experiment_data["adversarial_training"][key]["clean"]["losses"]["val"].append(
            vl_loss
        )
        experiment_data["adversarial_training"][key]["clean"]["metrics"][
            "orig_acc"
        ].append(orig_acc)
        experiment_data["adversarial_training"][key]["clean"]["metrics"][
            "robust_acc"
        ].append(robust_acc)
        print(
            f"[Adv Abl clean eps={eps}] Epoch {epoch}/{n_epochs} tr_loss:{tr_loss:.4f} val_loss:{vl_loss:.4f} orig_acc:{orig_acc:.4f} robust_acc:{robust_acc:.4f}"
        )
    _, _, pr, gt = evaluate(model_C, orig_test_loader, eps)
    experiment_data["adversarial_training"][key]["clean"]["predictions"] = pr
    experiment_data["adversarial_training"][key]["clean"]["ground_truth"] = gt

    # adversarial training
    model_A = CNN().to(device)
    opt_A = optim.Adam(model_A.parameters(), lr=1e-3)
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch_adv(model_A, opt_A, eps, epsilon_adv)
        vl_loss, orig_acc, _, _ = evaluate(model_A, orig_test_loader, eps)
        _, robust_acc, _, _ = evaluate_adv(model_A, orig_test_loader, epsilon_adv, eps)
        experiment_data["adversarial_training"][key]["adv"]["losses"]["train"].append(
            tr_loss
        )
        experiment_data["adversarial_training"][key]["adv"]["losses"]["val"].append(
            vl_loss
        )
        experiment_data["adversarial_training"][key]["adv"]["metrics"][
            "orig_acc"
        ].append(orig_acc)
        experiment_data["adversarial_training"][key]["adv"]["metrics"][
            "robust_acc"
        ].append(robust_acc)
        print(
            f"[Adv Abl adv eps={eps}] Epoch {epoch}/{n_epochs} tr_loss:{tr_loss:.4f} val_loss:{vl_loss:.4f} orig_acc:{orig_acc:.4f} robust_acc:{robust_acc:.4f}"
        )
    _, _, pr, gt = evaluate(model_A, orig_test_loader, eps)
    experiment_data["adversarial_training"][key]["adv"]["predictions"] = pr
    experiment_data["adversarial_training"][key]["adv"]["ground_truth"] = gt

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
