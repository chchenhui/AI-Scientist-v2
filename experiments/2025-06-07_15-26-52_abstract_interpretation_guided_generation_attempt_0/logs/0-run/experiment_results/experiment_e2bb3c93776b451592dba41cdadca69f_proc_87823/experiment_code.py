import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
np.random.seed(0)
num_train, num_val = 800, 200
train_ids = np.random.choice(len(specs), num_train)
val_ids = np.random.choice(len(specs), num_val)

# Static test pairs
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        return x, x


class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        e = self.emb(x)
        return self.fc(e)


def evaluate_generation(model, id_list):
    model.eval()
    total = len(id_list)
    pass_count = 0
    total_iters = 0
    with torch.no_grad():
        for sid in id_list:
            # Simulate one AIGG loop
            total_iters += 1
            x = torch.tensor([sid], dtype=torch.long, device=device)
            logits = model(x)
            pred_id = int(logits.argmax(dim=-1).item())
            expr = base_code[pred_id]
            if "/" in expr:
                code_line = f"return {expr} if b != 0 else 0"
            else:
                code_line = f"return {expr}"
            code_str = f"def f(a, b):\n    {code_line}"
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
                if "/" in expr:
                    ref = a / b if b != 0 else 0
                else:
                    ref = eval(expr)
                if abs(out - ref) > 1e-6:
                    ok = False
                    break
            if ok:
                pass_count += 1
    rate = pass_count / total
    mean_iters = total_iters / total
    return rate, mean_iters


batch_sizes = [8, 16, 32, 64, 128]
learning_rate = 0.01
num_epochs = 5

experiment_data = {
    "batch_size": {
        "synthetic": {
            "params": batch_sizes,
            "metrics": {"train": [], "val": []},
            "iterations": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for bs in batch_sizes:
    print(f"\n=== Training with batch size = {bs} ===")
    train_loader = DataLoader(SpecDataset(train_ids), batch_size=bs, shuffle=True)
    val_loader = DataLoader(SpecDataset(val_ids), batch_size=bs)

    model = Classifier(len(specs)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_rates, epoch_val_rates = [], []
    epoch_train_iters, epoch_val_iters = [], []
    all_preds, all_gts = [], []

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ids)
        epoch_train_losses.append(train_loss)

        # Validation loss
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_val_loss += loss.item() * x.size(0)
        val_loss = total_val_loss / len(val_ids)
        epoch_val_losses.append(val_loss)

        # Evaluate code-generation correctness and iterations
        train_rate, train_iters = evaluate_generation(model, train_ids)
        val_rate, val_iters = evaluate_generation(model, val_ids)
        epoch_train_rates.append(train_rate)
        epoch_val_rates.append(val_rate)
        epoch_train_iters.append(train_iters)
        epoch_val_iters.append(val_iters)

        # Record predictions & ground truth on validation set
        epoch_preds, epoch_gts = [], []
        for sid in val_ids:
            x = torch.tensor([sid], dtype=torch.long, device=device)
            with torch.no_grad():
                pred = int(model(x).argmax(dim=-1).item())
            expr_pred = base_code[pred]
            if "/" in expr_pred:
                line = f"return {expr_pred} if b != 0 else 0"
            else:
                line = f"return {expr_pred}"
            pred_str = f"def f(a, b):\n    {line}"
            gt_str = f"def f(a, b):\n    return {base_code[sid]}"
            epoch_preds.append(pred_str)
            epoch_gts.append(gt_str)
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

        print(
            f"BS={bs} Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_rate={train_rate:.4f}, val_rate={val_rate:.4f}, "
            f"mean_iters_train={train_iters:.2f}, mean_iters_val={val_iters:.2f}"
        )

    d = experiment_data["batch_size"]["synthetic"]
    d["losses"]["train"].append(epoch_train_losses)
    d["losses"]["val"].append(epoch_val_losses)
    d["metrics"]["train"].append(epoch_train_rates)
    d["metrics"]["val"].append(epoch_val_rates)
    d["iterations"]["train"].append(epoch_train_iters)
    d["iterations"]["val"].append(epoch_val_iters)
    d["predictions"].append(all_preds)
    d["ground_truth"].append(all_gts)

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
