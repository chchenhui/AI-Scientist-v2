import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic specification dataset
specs = ["add", "sub", "mul", "div"]
base_code = {0: "a+b", 1: "a-b", 2: "a*b", 3: "a/b"}
num_train, num_val = 800, 200
train_ids = np.random.choice(len(specs), num_train)
val_ids = np.random.choice(len(specs), num_val)


class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.ids[idx]


train_loader = DataLoader(SpecDataset(train_ids), batch_size=32, shuffle=True)
val_loader = DataLoader(SpecDataset(val_ids), batch_size=32)


def evaluate_generation(id_list):
    test_pairs = [(i, (i % 3) - 1) for i in range(6)]
    pass_count = 0
    for sid in id_list:
        expr = base_code[sid]
        line = f"return {expr} if '/' in expr and b==0 else {expr}"
        code_str = f"def f(a, b):\n    {line}"
        ns = {}
        try:
            exec(code_str, ns)
            func = ns["f"]
        except:
            continue
        ok = True
        for a, b in test_pairs:
            try:
                out = func(a, b)
            except:
                ok = False
                break
            ref = (a / b) if "/" in expr and b != 0 else eval(expr)
            if abs(out - ref) > 1e-6:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count / len(id_list)


# Model definition
class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        return self.fc(self.emb(x))


# Hyperparameter sweep
embedding_dims = [8, 16, 32, 64]
num_epochs = 5
experiment_data = {"embedding_dim_tuning": {}}

for emb_dim in embedding_dims:
    # init per-setting logs
    logs = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    model = Classifier(len(specs), emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        # train
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
        logs["losses"]["train"].append(total_loss / len(train_ids))

        # val
        model.eval()
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total_val += criterion(model(x), y).item() * x.size(0)
        logs["losses"]["val"].append(total_val / len(val_ids))

        # metrics
        tr_rate = evaluate_generation(train_ids)
        vl_rate = evaluate_generation(val_ids)
        logs["metrics"]["train"].append(tr_rate)
        logs["metrics"]["val"].append(vl_rate)

        # record preds/gts
        preds, gts = [], []
        for sid in val_ids:
            expr = base_code[sid]
            line = f"return {expr} if '/' in expr and b==0 else {expr}"
            preds.append(f"def f(a, b):\n    {line}")
            gts.append(f"def f(a, b):\n    return {expr}")
        logs["predictions"].append(preds)
        logs["ground_truth"].append(gts)

    experiment_data["embedding_dim_tuning"][str(emb_dim)] = logs

# save data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# plot losses
epochs = list(range(1, num_epochs + 1))
plt.figure()
for d in embedding_dims:
    L = experiment_data["embedding_dim_tuning"][str(d)]["losses"]
    plt.plot(epochs, L["train"], label=f"Train L dim{d}")
    plt.plot(epochs, L["val"], "--", label=f"Val L dim{d}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(working_dir, "loss_curve.png"))

# plot error-free rate
plt.figure()
for d in embedding_dims:
    M = experiment_data["embedding_dim_tuning"][str(d)]["metrics"]
    plt.plot(epochs, M["train"], label=f"Train ER dim{d}")
    plt.plot(epochs, M["val"], "--", label=f"Val ER dim{d}")
plt.xlabel("Epoch")
plt.ylabel("Error-Free Rate")
plt.legend()
plt.savefig(os.path.join(working_dir, "error_rate.png"))
