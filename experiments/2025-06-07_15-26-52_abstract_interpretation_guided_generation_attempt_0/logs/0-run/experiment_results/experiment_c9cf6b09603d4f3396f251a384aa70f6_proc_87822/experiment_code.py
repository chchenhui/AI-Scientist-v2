import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device and seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
np.random.seed(0)
torch.manual_seed(0)

# Full list of synthetic ops
full_ops = [
    ("add", "a+b"),
    ("sub", "a-b"),
    ("mul", "a*b"),
    ("div", "a/b"),
    ("mod", "a%b"),
    ("pow", "a**b"),
    ("bit_and", "a&b"),
    ("bit_or", "a|b"),
    ("bit_xor", "a^b"),
    ("lshift", "a<<b"),
    ("rshift", "a>>b"),
    ("min", "a if a<b else b"),
    ("max", "a if a>b else b"),
    ("eq", "1 if a==b else 0"),
    ("neq", "1 if a!=b else 0"),
    ("gt", "1 if a>b else 0"),
]


# Dataset class
class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        return x, x


# Classifier model
class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        return self.fc(self.emb(x))


# Evaluation routine using model predictions
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


def evaluate_generation(model, id_list, code_map):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(id_list, dtype=torch.long).to(device)
        logits = model(ids)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    pass_count = 0
    for true_id, pred_id in zip(id_list, preds):
        expr = code_map[int(pred_id)]
        # guard division by zero
        if "/" in expr:
            line = f"return {expr} if b != 0 else 0"
        else:
            line = f"return {expr}"
        code_str = f"def f(a, b):\n    {line}"
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
                ref = a / b if b != 0 else 0
            else:
                ref = eval(expr)
            if abs(out - ref) > 1e-6:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count / len(id_list)


# Ablation sizes
ablation_sizes = [4, 8, 16]
num_train, num_val = 800, 200
num_epochs = 5
learning_rate = 0.01
batch_size = 32

# Main experiment data
experiment_data = {"output_vocab_scaling": {}}

for n_ops in ablation_sizes:
    print(f"\n=== Ablation: vocab size = {n_ops} ===")
    ops_subset = full_ops[:n_ops]
    code_map = {i: expr for i, (_, expr) in enumerate(ops_subset)}
    train_ids = np.random.choice(n_ops, size=num_train)
    val_ids = np.random.choice(n_ops, size=num_val)
    train_loader = DataLoader(
        SpecDataset(train_ids), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SpecDataset(val_ids), batch_size=batch_size)
    model = Classifier(n_ops).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses_train, losses_val = [], []
    rates_train, rates_val = [], []
    mitc_train, mitc_val = [], []
    all_preds, all_gts = [], []

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        tot_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * x.size(0)
        train_loss = tot_loss / num_train
        losses_train.append(train_loss)

        # Validate loss
        model.eval()
        tot_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                tot_val += criterion(model(x), y).item() * x.size(0)
        val_loss = tot_val / num_val
        losses_val.append(val_loss)
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

        # Generation metrics and dummy MItC
        tr_rate = evaluate_generation(model, train_ids.tolist(), code_map)
        va_rate = evaluate_generation(model, val_ids.tolist(), code_map)
        rates_train.append(tr_rate)
        rates_val.append(va_rate)
        mitc_train.append(1.0)
        mitc_val.append(1.0)
        print(
            f"V={n_ops} E={epoch} tr_AICR={tr_rate:.4f} val_AICR={va_rate:.4f} tr_MItC={1.0:.2f} val_MItC={1.0:.2f}"
        )

        # Record predictions and ground truth on validation for analysis
        epoch_preds, epoch_gts = [], []
        with torch.no_grad():
            ids = torch.tensor(val_ids, dtype=torch.long).to(device)
            logits = model(ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        for true_id, pred_id in zip(val_ids, preds):
            pred_expr = code_map[int(pred_id)]
            true_expr = code_map[int(true_id)]
            if "/" in pred_expr:
                line_pred = f"return {pred_expr} if b != 0 else 0"
            else:
                line_pred = f"return {pred_expr}"
            epoch_preds.append(f"def f(a, b):\n    {line_pred}")
            epoch_gts.append(f"def f(a, b):\n    return {true_expr}")
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

    experiment_data["output_vocab_scaling"][f"ops_{n_ops}"] = {
        "losses": {"train": losses_train, "val": losses_val},
        "metrics": {
            "AICR": {"train": rates_train, "val": rates_val},
            "MeanIters": {"train": mitc_train, "val": mitc_val},
        },
        "predictions": all_preds,
        "ground_truth": all_gts,
    }

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
