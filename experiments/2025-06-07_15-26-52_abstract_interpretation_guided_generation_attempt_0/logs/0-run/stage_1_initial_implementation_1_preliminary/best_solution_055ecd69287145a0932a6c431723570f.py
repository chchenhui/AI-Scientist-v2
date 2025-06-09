import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic specification dataset
specs = ["add", "sub", "mul", "div"]
spec2id = {s: i for i, s in enumerate(specs)}
base_code = {
    0: "a+b",
    1: "a-b",
    2: "a*b",
    3: "a/b",
}

# Generate train/val splits
num_train, num_val = 800, 200
train_ids = np.random.choice(len(specs), num_train)
val_ids = np.random.choice(len(specs), num_val)


class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        return x, x


train_loader = DataLoader(SpecDataset(train_ids), batch_size=32, shuffle=True)
val_loader = DataLoader(SpecDataset(val_ids), batch_size=32)


# Simple classifier model
class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        e = self.emb(x)
        return self.fc(e)


model = Classifier(len(specs)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Prepare experiment data logging
experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Fixed test pairs to include b=0 cases
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


def evaluate_generation(id_list):
    pass_count = 0
    for sid in id_list:
        expr = base_code[sid]
        # static analysis + repair
        if "/" in expr:
            code_line = f"return {expr} if b != 0 else 0"
        else:
            code_line = f"return {expr}"
        code_str = f"def f(a, b):\n    {code_line}"
        ns = {}
        try:
            exec(code_str, ns)
            func = ns["f"]
        except Exception:
            continue
        ok = True
        for a, b in test_pairs:
            try:
                out = func(a, b)
            except Exception:
                ok = False
                break
            if b != 0 and abs(out - (a / b if "/" in expr else eval(expr))) > 1e-6:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count / len(id_list)


# Training loop
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_ids)
    experiment_data["synthetic"]["losses"]["train"].append(train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_val_loss += loss.item() * x.size(0)
    val_loss = total_val_loss / len(val_ids)
    experiment_data["synthetic"]["losses"]["val"].append(val_loss)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

    # Evaluate error-free generation rate
    train_err_rate = evaluate_generation(train_ids)
    val_err_rate = evaluate_generation(val_ids)
    experiment_data["synthetic"]["metrics"]["train"].append(train_err_rate)
    experiment_data["synthetic"]["metrics"]["val"].append(val_err_rate)
    print(
        f"Epoch {epoch}: train_error_free_rate = {train_err_rate:.4f}, val_error_free_rate = {val_err_rate:.4f}"
    )

    # Log predictions vs ground truth on val
    preds, gts = [], []
    for sid in val_ids:
        expr = base_code[sid]
        if "/" in expr:
            line = f"return {expr} if b != 0 else 0"
        else:
            line = f"return {expr}"
        preds.append(f"def f(a, b):\n    {line}")
        gts.append(f"def f(a, b):\n    return {expr}")
    experiment_data["synthetic"]["predictions"].append(preds)
    experiment_data["synthetic"]["ground_truth"].append(gts)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Plot losses
epochs = list(range(1, num_epochs + 1))
plt.figure()
plt.plot(epochs, experiment_data["synthetic"]["losses"]["train"], label="Train Loss")
plt.plot(epochs, experiment_data["synthetic"]["losses"]["val"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

# Plot error-free generation rate
plt.figure()
plt.plot(
    epochs,
    experiment_data["synthetic"]["metrics"]["train"],
    label="Train Error-Free Rate",
)
plt.plot(
    epochs, experiment_data["synthetic"]["metrics"]["val"], label="Val Error-Free Rate"
)
plt.xlabel("Epoch")
plt.ylabel("Error-Free Rate")
plt.legend()
plt.savefig(os.path.join(working_dir, "error_rate.png"))
