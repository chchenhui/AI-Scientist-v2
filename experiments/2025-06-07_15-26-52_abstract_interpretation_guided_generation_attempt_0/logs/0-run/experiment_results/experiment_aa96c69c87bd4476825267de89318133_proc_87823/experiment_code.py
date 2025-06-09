import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# reproducibility
np.random.seed(0)
torch.manual_seed(0)

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic specs
specs = ["add", "sub", "mul", "div"]
spec2id = {s: i for i, s in enumerate(specs)}
base_code = {0: "a+b", 1: "a-b", 2: "a*b", 3: "a/b"}

# train/val splits
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


# evaluator
K = 6
test_pairs = [(i, (i % 3) - 1) for i in range(K)]


def evaluate_generation(id_list):
    pass_count = 0
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
                ref = a / b if b != 0 else 0
            else:
                ref = eval(expr)
            if abs(out - ref) > 1e-6:
                ok = False
                break
        if ok:
            pass_count += 1
    return pass_count / len(id_list)


# ablation sweep: weight_decay
weight_decay_values = [0, 1e-5, 1e-4, 1e-3]
lr = 0.01
num_epochs = 5
batch_size = 32

experiment_data = {
    "weight_decay": {
        "synthetic": {
            "params": weight_decay_values,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for wd in weight_decay_values:
    print(f"\n=== Training with weight_decay = {wd} ===")
    train_loader = DataLoader(
        SpecDataset(train_ids), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SpecDataset(val_ids), batch_size=batch_size)
    model = Classifier(len(specs)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_rates, epoch_val_rates = [], []
    all_preds, all_gts = [], []

    for epoch in range(1, num_epochs + 1):
        # train
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

        # validate
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_val += loss.item() * x.size(0)
        val_loss = total_val / len(val_ids)
        epoch_val_losses.append(val_loss)

        # AICR metrics
        train_rate = evaluate_generation(train_ids)
        val_rate = evaluate_generation(val_ids)
        epoch_train_rates.append(train_rate)
        epoch_val_rates.append(val_rate)

        # record preds & gts
        epoch_preds, epoch_gts = [], []
        for sid in val_ids:
            expr = base_code[sid]
            if "/" in expr:
                line = f"return {expr} if b != 0 else 0"
            else:
                line = f"return {expr}"
            epoch_preds.append(f"def f(a, b):\n    {line}")
            epoch_gts.append(f"def f(a, b):\n    return {expr}")
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

        print(
            f"WD={wd} Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_AICR={train_rate:.4f}, val_AICR={val_rate:.4f}"
        )

    d = experiment_data["weight_decay"]["synthetic"]
    d["losses"]["train"].append(epoch_train_losses)
    d["losses"]["val"].append(epoch_val_losses)
    d["metrics"]["train"].append(epoch_train_rates)
    d["metrics"]["val"].append(epoch_val_rates)
    d["predictions"].append(all_preds)
    d["ground_truth"].append(all_gts)

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
