import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
import numpy as np


class CNN(nn.Module):
    def __init__(self, in_ch, num_classes, hidden1, hidden2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, hidden1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden1, hidden2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"] if "image" in item else item["img"]
        image = self.transform(image)
        label = item["label"]
        return image, label


def train_one_epoch(model, optimizer, loader):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    total, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += preds.eq(y).sum().item()
    return total / len(loader.dataset), correct / len(loader.dataset)


dataset_names = ["mnist", "cifar10", "fashion_mnist"]
dataset_stats = {
    "mnist": {"mean": (0.1307,), "std": (0.3081,)},
    "fashion_mnist": {"mean": (0.2860,), "std": (0.3530,)},
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
}

experiment_data = {}
n_epochs = 3
batch_size = 128

for ds in dataset_names:
    ds_hf = load_dataset(ds)
    stats = dataset_stats[ds]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])]
    )
    train_ds = HFImageDataset(ds_hf["train"], transform)
    test_ds = HFImageDataset(ds_hf["test"], transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    in_ch = 3 if ds == "cifar10" else 1
    num_cls = len(np.unique(np.array(ds_hf["train"]["label"])))
    model_configs = [(16, 32), (32, 64), (64, 128)]
    models, opts = [], []
    for h1, h2 in model_configs:
        m = CNN(in_ch, num_cls, h1, h2).to(device)
        models.append(m)
        opts.append(optim.Adam(m.parameters(), lr=1e-3))

    experiment_data[ds] = {"losses": {"train": [], "val": []}, "disc_score": []}
    for epoch in range(1, n_epochs + 1):
        train_losses, val_losses, accs = [], [], []
        for model, opt in zip(models, opts):
            tr = train_one_epoch(model, opt, train_loader)
            vl, acc = evaluate(model, test_loader)
            train_losses.append(tr)
            val_losses.append(vl)
            accs.append(acc)
        avg_tr = float(np.mean(train_losses))
        avg_vl = float(np.mean(val_losses))
        disc = float(np.std(accs))
        experiment_data[ds]["losses"]["train"].append(avg_tr)
        experiment_data[ds]["losses"]["val"].append(avg_vl)
        experiment_data[ds]["disc_score"].append(disc)
        print(
            f"[{ds}] Epoch {epoch}: validation_loss = {avg_vl:.4f}, Benchmark Discrimination Score = {disc:.4f}"
        )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
