import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
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


class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        return x, x


class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16, dropout_rate=0.0):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        e = self.emb(x)
        e = self.dropout(e)
        return self.fc(e)


# Test pairs for functional correctness
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


def evaluate_generation(model, id_list):
    model.eval()
    pass_count = 0
    with torch.no_grad():
        for sid in id_list:
            x = torch.tensor([sid], dtype=torch.long).to(device)
            logits = model(x)
            y_pred = logits.argmax(dim=1).item()
            expr_pred = base_code[y_pred]
            expr_gt = base_code[sid]
            # generate safe code for predicted op
            if "/" in expr_pred:
                code_line = f"return {expr_pred} if b != 0 else 0"
            else:
                code_line = f"return {expr_pred}"
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
                # compute ground-truth reference
                if "/" in expr_gt:
                    ref = a / b if b != 0 else 0
                else:
                    ref = eval(expr_gt)
                if abs(out - ref) > 1e-6:
                    ok = False
                    break
            if ok:
                pass_count += 1
    return pass_count / len(id_list)


# Ablation study parameters
dropout_rates = [0.0, 0.2, 0.5]
learning_rate = 0.005
num_epochs = 5

experiment_data = {
    "dropout_ablation": {
        "synthetic": {
            "params": dropout_rates,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for dr in dropout_rates:
    print(f"\n=== Training with dropout_rate = {dr} ===")
    train_loader = DataLoader(SpecDataset(train_ids), batch_size=32, shuffle=True)
    val_loader = DataLoader(SpecDataset(val_ids), batch_size=32)

    model = Classifier(len(specs), emb_dim=16, dropout_rate=dr).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_rates, epoch_val_rates = [], []
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
                total_val_loss += criterion(logits, y).item() * x.size(0)
        val_loss = total_val_loss / len(val_ids)
        epoch_val_losses.append(val_loss)

        # Evaluate generation rates
        train_rate = evaluate_generation(model, train_ids)
        val_rate = evaluate_generation(model, val_ids)
        epoch_train_rates.append(train_rate)
        epoch_val_rates.append(val_rate)

        # Record predictions & ground truth
        epoch_preds, epoch_gts = [], []
        with torch.no_grad():
            for sid in val_ids:
                x = torch.tensor([sid], dtype=torch.long).to(device)
                y_pred = model(x).argmax(dim=1).item()
                expr_pred = base_code[y_pred]
                if "/" in expr_pred:
                    pred_line = f"return {expr_pred} if b != 0 else 0"
                else:
                    pred_line = f"return {expr_pred}"
                epoch_preds.append(f"def f(a, b):\n    {pred_line}")
                epoch_gts.append(f"def f(a, b):\n    return {base_code[sid]}")
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

        print(
            f"DR={dr} Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_rate={train_rate:.4f}, val_rate={val_rate:.4f}"
        )

    d = experiment_data["dropout_ablation"]["synthetic"]
    d["losses"]["train"].append(epoch_train_losses)
    d["losses"]["val"].append(epoch_val_losses)
    d["metrics"]["train"].append(epoch_train_rates)
    d["metrics"]["val"].append(epoch_val_rates)
    d["predictions"].append(all_preds)
    d["ground_truth"].append(all_gts)

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
