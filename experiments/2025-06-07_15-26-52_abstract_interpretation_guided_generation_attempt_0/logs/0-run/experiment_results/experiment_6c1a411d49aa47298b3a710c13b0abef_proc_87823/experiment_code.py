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

# Seeds
np.random.seed(0)
torch.manual_seed(0)


# Dataset
class SpecDataset(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        return x, x


# Model
class Classifier(nn.Module):
    def __init__(self, n_ops, emb_dim=16):
        super().__init__()
        self.emb = nn.Embedding(n_ops, emb_dim)
        self.fc = nn.Linear(emb_dim, n_ops)

    def forward(self, x):
        return self.fc(self.emb(x))


# Generation evaluator (uses model predictions now)
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


def evaluate_generation(model, id_list, base_code):
    model.eval()
    ids = torch.tensor(id_list, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids)
        preds = logits.argmax(dim=1).cpu().tolist()
    pass_count = 0
    for psid in preds:
        expr = base_code[psid]
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
                ref = a / b if b != 0 else 0
            else:
                ref = eval(expr)
            if abs(out - ref) > 1e-6:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count / len(preds)


# Settings
num_train, num_val = 800, 200
num_epochs = 5
batch_size = 32
learning_rate = 0.01

# Domain configurations
domain_configs = {
    "arithmetic": {
        "specs": ["add", "sub", "mul", "div"],
        "base_code": {0: "a+b", 1: "a-b", 2: "a*b", 3: "a/b"},
    },
    "polynomial": {
        "specs": ["poly2", "poly1"],
        "base_code": {0: "a*a + b*a + 1", 1: "a*b + 1"},
    },
    "bitwise": {
        "specs": ["and", "or", "xor", "shl", "shr"],
        "base_code": {0: "a & b", 1: "a | b", 2: "a ^ b", 3: "a << b", 4: "a >> b"},
    },
}
# Combined domain
combined_specs = (
    domain_configs["arithmetic"]["specs"]
    + domain_configs["polynomial"]["specs"]
    + domain_configs["bitwise"]["specs"]
)
combined_base_code = {}
idx = 0
for ds in ["arithmetic", "polynomial", "bitwise"]:
    codes = [
        domain_configs[ds]["base_code"][i]
        for i in range(len(domain_configs[ds]["specs"]))
    ]
    for code in codes:
        combined_base_code[idx] = code
        idx += 1
domain_configs["combined"] = {"specs": combined_specs, "base_code": combined_base_code}

# Prepare experiment_data
experiment_data = {"multi_domain_synthetic_specification": {}}
for ds in domain_configs:
    experiment_data["multi_domain_synthetic_specification"][ds] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

# Run experiments
for ds, config in domain_configs.items():
    print(f"\n=== Dataset: {ds} ===")
    specs = config["specs"]
    base_code = config["base_code"]
    n_ops = len(specs)
    # Splits
    train_ids = np.random.choice(n_ops, num_train)
    val_ids = np.random.choice(n_ops, num_val)
    train_list = train_ids.tolist()
    val_list = val_ids.tolist()
    train_loader = DataLoader(
        SpecDataset(train_ids), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SpecDataset(val_ids), batch_size=batch_size)
    # Model
    model = Classifier(n_ops).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Stats
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_rates, epoch_val_rates = [], []
    all_preds, all_gts = [], []
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        tot = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item() * x.size(0)
        train_loss = tot / num_train
        epoch_train_losses.append(train_loss)
        # Val loss
        model.eval()
        totv = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                totv += loss.item() * x.size(0)
        val_loss = totv / num_val
        epoch_val_losses.append(val_loss)
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        # Rates using model predictions
        tr_rate = evaluate_generation(model, train_list, base_code)
        vl_rate = evaluate_generation(model, val_list, base_code)
        epoch_train_rates.append(tr_rate)
        epoch_val_rates.append(vl_rate)
        # Predictions & GT for storage
        preds, gts = [], []
        for sid in val_list:
            # use model to predict
            with torch.no_grad():
                inp = torch.tensor([sid], device=device)
                pred = model(inp).argmax(dim=1).item()
            expr_p = base_code[pred]
            expr_gt = base_code[sid]
            pl = f"return {expr_p} if '/' in '{expr_p}' and b == 0 else {expr_p}"
            preds.append(f"def f(a, b):\n    {pl}")
            gts.append(f"def f(a, b):\n    return {expr_gt}")
        all_preds.append(preds)
        all_gts.append(gts)
        print(
            f"Dataset={ds} Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_rate={tr_rate:.4f}, val_rate={vl_rate:.4f}"
        )
    # Store
    entry = experiment_data["multi_domain_synthetic_specification"][ds]
    entry["losses"]["train"] = epoch_train_losses
    entry["losses"]["val"] = epoch_val_losses
    entry["metrics"]["train"] = epoch_train_rates
    entry["metrics"]["val"] = epoch_val_rates
    entry["predictions"] = all_preds
    entry["ground_truth"] = all_gts

# Save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
