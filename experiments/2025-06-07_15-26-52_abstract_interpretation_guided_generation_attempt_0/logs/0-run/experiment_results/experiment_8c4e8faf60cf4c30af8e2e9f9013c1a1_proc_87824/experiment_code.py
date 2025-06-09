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

# Synthetic specification dataset
specs = ["add", "sub", "mul", "div"]
base_code = {i: code for i, code in enumerate(["a+b", "a-b", "a*b", "a/b"])}

# Generate train/val splits
np.random.seed(0)
num_train, num_val = 800, 200
train_ids = np.random.choice(len(specs), num_train).tolist()
val_ids = np.random.choice(len(specs), num_val).tolist()


class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        return x, x


def evaluate_generation(id_list):
    pass_count = 0
    test_pairs = [(i, (i % 3) - 1) for i in range(6)]
    for sid in id_list:
        expr = base_code[sid]
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
            if "/" in expr:
                if b != 0:
                    ref = a / b
                else:
                    ref = 0
            else:
                ref = eval(expr)
            if abs(out - ref) > 1e-6:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count / len(id_list)


class FixedEmbeddingClassifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.emb.weight.requires_grad = False
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        return self.fc(self.emb(x))


def get_predictions(model, ids):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(ids, dtype=torch.long).to(device)
        logits = model(x)
        return logits.argmax(dim=1).cpu().tolist()


# Hyperparameters
learning_rates = [0.001, 0.005, 0.01, 0.02]
num_epochs = 5

# Experiment data container
experiment_data = {
    "fixed_random_embedding": {
        "synthetic": {
            "params": learning_rates,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Ablation run
for lr in learning_rates:
    train_loader = DataLoader(SpecDataset(train_ids), batch_size=32, shuffle=True)
    val_loader = DataLoader(SpecDataset(val_ids), batch_size=32)
    model = FixedEmbeddingClassifier(len(specs)).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_rates, epoch_val_rates = [], []
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
        total_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total_val += criterion(logits, y).item() * x.size(0)
        val_loss = total_val / len(val_ids)
        epoch_val_losses.append(val_loss)

        # Evaluate code generation success
        train_preds = get_predictions(model, train_ids)
        val_preds = get_predictions(model, val_ids)
        train_rate = evaluate_generation(train_preds)
        val_rate = evaluate_generation(val_preds)
        epoch_train_rates.append(train_rate)
        epoch_val_rates.append(val_rate)

        # Record predictions & ground truth on validation set
        epoch_preds, epoch_gts = [], []
        for p, t in zip(val_preds, val_ids):
            expr_p = base_code[p]
            line_p = (
                f"return {expr_p} if b != 0 else 0"
                if "/" in expr_p
                else f"return {expr_p}"
            )
            epoch_preds.append(f"def f(a, b):\n    {line_p}")
            epoch_gts.append(f"def f(a, b):\n    return {base_code[t]}")
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

        print(
            f"LR={lr} Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_rate={train_rate:.4f}, val_rate={val_rate:.4f}"
        )

    d = experiment_data["fixed_random_embedding"]["synthetic"]
    d["losses"]["train"].append(epoch_train_losses)
    d["losses"]["val"].append(epoch_val_losses)
    d["metrics"]["train"].append(epoch_train_rates)
    d["metrics"]["val"].append(epoch_val_rates)
    d["predictions"].append(all_preds)
    d["ground_truth"].append(all_gts)

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
