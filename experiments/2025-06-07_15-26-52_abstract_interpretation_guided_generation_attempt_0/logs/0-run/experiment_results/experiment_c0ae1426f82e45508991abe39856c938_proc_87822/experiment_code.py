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
base_code = {0: "a+b", 1: "a-b", 2: "a*b", 3: "a/b"}

# Generate train/val splits
np.random.seed(0)
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


class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        return self.fc(self.emb(x))


# Generator evaluator now uses the model's predictions
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


def evaluate_generation(model, id_list):
    model.eval()
    pass_count = 0
    iter_counts = []
    with torch.no_grad():
        for sid in id_list:
            x = torch.tensor([sid], dtype=torch.long).to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1).item()
            expr = base_code[pred]
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
                iter_counts.append(1)
                continue
            ok = True
            for a, b in test_pairs:
                try:
                    out = func(a, b)
                except Exception:
                    ok = False
                    break
                ref = (
                    a / b
                    if "/" in expr and b != 0
                    else (0 if "/" in expr else eval(expr))
                )
                if abs(out - ref) > 1e-6:
                    ok = False
                    break
            if ok:
                pass_count += 1
            iter_counts.append(1)
    pass_rate = pass_count / len(id_list)
    mean_iters = float(np.mean(iter_counts)) if iter_counts else 0.0
    return pass_rate, mean_iters


# Hyperparameters for ablation
smoothing_factors = [0.0, 0.1, 0.2]
learning_rate = 0.01
num_epochs = 5
batch_size = 32

# Initialize experiment_data
experiment_data = {
    "label_smoothing": {
        "synthetic": {
            "params": smoothing_factors,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "mean_iterations": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# DataLoaders
train_loader = DataLoader(SpecDataset(train_ids), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SpecDataset(val_ids), batch_size=batch_size)

# Ablation over smoothing factors
for smooth in smoothing_factors:
    print(f"\n=== Label smoothing = {smooth} ===")
    model = Classifier(len(specs)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=smooth)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_rates, epoch_val_rates = [], []
    epoch_train_iters, epoch_val_iters = [], []
    all_preds, all_gts = [], []

    for epoch in range(1, num_epochs + 1):
        # Train
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

        # Validate loss
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
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

        # Evaluate generation pass rates and mean iters
        train_rate, train_mean_iters = evaluate_generation(model, train_ids)
        val_rate, val_mean_iters = evaluate_generation(model, val_ids)
        epoch_train_rates.append(train_rate)
        epoch_val_rates.append(val_rate)
        epoch_train_iters.append(train_mean_iters)
        epoch_val_iters.append(val_mean_iters)

        # Collect predicted vs. ground-truth code strings
        epoch_preds, epoch_gts = [], []
        with torch.no_grad():
            for sid in val_ids:
                x = torch.tensor([sid], dtype=torch.long).to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1).item()
                expr_pred = base_code[pred]
                if "/" in expr_pred:
                    line_p = f"return {expr_pred} if b != 0 else 0"
                else:
                    line_p = f"return {expr_pred}"
                epoch_preds.append(f"def f(a, b):\n    {line_p}")
                expr_true = base_code[int(sid)]
                epoch_gts.append(f"def f(a, b):\n    return {expr_true}")
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

        print(
            f"smooth={smooth} Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_rate(AICR)={train_rate:.4f}, val_rate(AICR)={val_rate:.4f}, "
            f"mean_iters_train={train_mean_iters:.2f}, mean_iters_val={val_mean_iters:.2f}"
        )

    # Append results
    d = experiment_data["label_smoothing"]["synthetic"]
    d["losses"]["train"].append(epoch_train_losses)
    d["losses"]["val"].append(epoch_val_losses)
    d["metrics"]["train"].append(epoch_train_rates)
    d["metrics"]["val"].append(epoch_val_rates)
    d["mean_iterations"]["train"].append(epoch_train_iters)
    d["mean_iterations"]["val"].append(epoch_val_iters)
    d["predictions"].append(all_preds)
    d["ground_truth"].append(all_gts)

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
