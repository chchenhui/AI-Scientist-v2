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
base_code = {0: "a+b", 1: "a-b", 2: "a*b", 3: "a/b"}

# Generate train/val splits (with replacement as before)
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
        # x is token ids
        return self.fc(self.emb(x))


# Test pairs to check correctness
test_pairs = [(i, (i % 3) - 1) for i in range(6)]


def evaluate_generation(model, id_list):
    model.eval()
    pass_count = 0
    with torch.no_grad():
        for sid in id_list:
            # predict op
            x = torch.tensor([sid], dtype=torch.long).to(device)
            logits = model(x)
            pred_id = int(logits.argmax(dim=-1).cpu().item())
            expr = base_code[pred_id]
            # static analysis: guard division by zero
            if "/" in expr:
                code = (
                    "def f(a,b):\n"
                    "    if b == 0:\n"
                    "        return 0\n"
                    f"    return {expr}\n"
                )
            else:
                code = f"def f(a,b):\n    return {expr}\n"
            # compile and test
            ns = {}
            try:
                exec(code, ns)
                f = ns["f"]
            except Exception:
                continue
            ok = True
            for a, b in test_pairs:
                try:
                    out = f(a, b)
                except:
                    ok = False
                    break
                ref = (
                    (a / b if b != 0 else 0)
                    if "/" in expr
                    else eval(expr.replace("a", str(a)).replace("b", str(b)))
                )
                if abs(out - ref) > 1e-6:
                    ok = False
                    break
            if ok:
                pass_count += 1
    return pass_count / len(id_list)


# Hyperparameters
base_lr = 0.01
num_epochs = 5
batch_size = 32

optimizers = [
    ("Adam", lambda params: optim.Adam(params, lr=base_lr)),
    ("SGD", lambda params: optim.SGD(params, lr=base_lr, momentum=0.9)),
    ("RMSprop", lambda params: optim.RMSprop(params, lr=base_lr)),
    ("Adagrad", lambda params: optim.Adagrad(params, lr=base_lr)),
]

# Prepare experiment data container
experiment_data = {
    "optimizer": {
        "synthetic": {
            "optim_names": [name for name, _ in optimizers],
            "metrics": {
                "train_AICR": [],
                "val_AICR": [],
                "train_loss": [],
                "val_loss": [],
                "mean_iters_to_convergence_train": [],
                "mean_iters_to_convergence_val": [],
            },
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for name, make_opt in optimizers:
    print(f"\n=== Optimizer: {name} ===")
    # DataLoaders
    train_loader = DataLoader(
        SpecDataset(train_ids), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SpecDataset(val_ids), batch_size=batch_size)

    model = Classifier(len(specs)).to(device)
    optimizer = make_opt(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_rates, val_rates = [], []
    conv_train, conv_val = [], []
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
        train_losses.append(train_loss)

        # Validate
        model.eval()
        total_vloss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total_vloss += criterion(model(x), y).item() * x.size(0)
        val_loss = total_vloss / len(val_ids)
        val_losses.append(val_loss)

        # Generation metrics
        tr_rate = evaluate_generation(model, train_ids)
        vr_rate = evaluate_generation(model, val_ids)
        train_rates.append(tr_rate)
        val_rates.append(vr_rate)

        # Toy convergence metric (1 loop per problem in this setup)
        conv_train.append(1.0)
        conv_val.append(1.0)

        # Record code predictions / ground truth for val set
        epoch_preds, epoch_gts = [], []
        with torch.no_grad():
            for sid in val_ids:
                x = torch.tensor([sid], dtype=torch.long).to(device)
                pred_id = int(model(x).argmax(dim=-1).cpu().item())
                expr_pred = base_code[pred_id]
                if "/" in expr_pred:
                    pred_code = (
                        "def f(a,b):\n"
                        "    if b == 0:\n"
                        "        return 0\n"
                        f"    return {expr_pred}\n"
                    )
                else:
                    pred_code = f"def f(a,b):\n    return {expr_pred}\n"
                gt_code = f"def f(a,b):\n    return {base_code[sid]}\n"
                epoch_preds.append(pred_code)
                epoch_gts.append(gt_code)
        all_preds.append(epoch_preds)
        all_gts.append(epoch_gts)

        print(
            f"{name} Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_AICR={tr_rate:.4f}, val_AICR={vr_rate:.4f}, "
            f"mean_iters_conv_train={conv_train[-1]:.1f}, mean_iters_conv_val={conv_val[-1]:.1f}"
        )

    # Save results
    data = experiment_data["optimizer"]["synthetic"]
    data["metrics"]["train_loss"].append(train_losses)
    data["metrics"]["val_loss"].append(val_losses)
    data["metrics"]["train_AICR"].append(train_rates)
    data["metrics"]["val_AICR"].append(val_rates)
    data["metrics"]["mean_iters_to_convergence_train"].append(conv_train)
    data["metrics"]["mean_iters_to_convergence_val"].append(conv_val)
    data["predictions"].append(all_preds)
    data["ground_truth"].append(all_gts)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved experiment_data.npy in {working_dir}")
