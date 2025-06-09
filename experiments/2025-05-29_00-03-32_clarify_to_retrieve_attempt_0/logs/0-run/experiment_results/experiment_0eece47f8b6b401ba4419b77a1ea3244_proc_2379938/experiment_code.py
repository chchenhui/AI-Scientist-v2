# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic XOR data
def make_xor(n):
    X = np.random.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


train_X, train_y = make_xor(2000)
val_X, val_y = make_xor(500)
train_ds = TensorDataset(train_X, train_y)
val_ds = TensorDataset(val_X, val_y)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


# MLP with variable hidden size
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# Hyperparameters
hidden_layer_sizes = [4, 8, 16, 32, 64]
epochs = 10
mc_T = 5
threshold = 0.02

# Experiment data container
experiment_data = {
    "hidden_layer_size": {
        "synthetic_xor": {
            "sizes": hidden_layer_sizes,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Loop over hidden sizes
for size in hidden_layer_sizes:
    print(f"\nStarting training with hidden_layer_size = {size}")
    # Initialize storage for this hyperparam
    mets_tr, mets_val = [], []
    losses_tr, losses_val = [], []
    preds_all, gts_all = [], []

    # Model, loss, optimizer
    model = MLP(size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training epochs
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, total_corr = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
            total_corr += (out.argmax(1) == yb).sum().item()
        train_loss = total_loss / len(train_ds)
        losses_tr.append(train_loss)

        # CES on train
        model.eval()
        base_corr, clar_corr, clar_count = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                X_mask = Xb.clone()
                X_mask[:, 1] = 0
                out_base = model(X_mask)
                preds_base = out_base.argmax(1)
                base_corr += (preds_base == yb).sum().item()
                for i in range(Xb.size(0)):
                    xi = X_mask[i : i + 1]
                    model.train()
                    ps = []
                    for _ in range(mc_T):
                        p = torch.softmax(model(xi), dim=1)
                        ps.append(p.cpu().numpy())
                    ps = np.stack(ps, 0)
                    model.eval()
                    var = ps.var(0).sum()
                    if var > threshold:
                        clar_count += 1
                        out_c = model(Xb[i : i + 1])
                        clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                    else:
                        clar_corr += (preds_base[i] == yb[i]).item()
        base_acc_tr = base_corr / len(train_ds)
        clar_acc_tr = clar_corr / len(train_ds)
        avg_ct_tr = clar_count / len(train_ds) if len(train_ds) else 0
        ces_tr = (clar_acc_tr - base_acc_tr) / avg_ct_tr if avg_ct_tr > 0 else 0.0
        mets_tr.append(ces_tr)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item() * Xb.size(0)
        val_loss /= len(val_ds)
        losses_val.append(val_loss)

        # CES on val
        base_corr, clar_corr, clar_count = 0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                X_mask = Xb.clone()
                X_mask[:, 1] = 0
                out_base = model(X_mask)
                preds_base = out_base.argmax(1)
                base_corr += (preds_base == yb).sum().item()
                for i in range(Xb.size(0)):
                    xi = X_mask[i : i + 1]
                    model.train()
                    ps = []
                    for _ in range(mc_T):
                        p = torch.softmax(model(xi), dim=1)
                        ps.append(p.cpu().numpy())
                    ps = np.stack(ps, 0)
                    model.eval()
                    var = ps.var(0).sum()
                    if var > threshold:
                        clar_count += 1
                        out_c = model(Xb[i : i + 1])
                        clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                    else:
                        clar_corr += (preds_base[i] == yb[i]).item()
        base_acc_val = base_corr / len(val_ds)
        clar_acc_val = clar_corr / len(val_ds)
        avg_ct_val = clar_count / len(val_ds) if len(val_ds) else 0
        ces_val = (clar_acc_val - base_acc_val) / avg_ct_val if avg_ct_val > 0 else 0.0
        mets_val.append(ces_val)

        # Store predictions & ground truth
        preds_list, gts_list = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                preds_list.append(out.argmax(1).cpu().numpy())
                gts_list.append(yb.cpu().numpy())
        preds_all.append(np.concatenate(preds_list))
        gts_all.append(np.concatenate(gts_list))

        print(
            f"Size {size}, Epoch {epoch}: train_loss={train_loss:.4f}, train_CES={ces_tr:.4f}, val_loss={val_loss:.4f}, val_CES={ces_val:.4f}"
        )

    # Append results for this hidden size
    exp = experiment_data["hidden_layer_size"]["synthetic_xor"]
    exp["metrics"]["train"].append(mets_tr)
    exp["metrics"]["val"].append(mets_val)
    exp["losses"]["train"].append(losses_tr)
    exp["losses"]["val"].append(losses_val)
    exp["predictions"].append(preds_all)
    exp["ground_truth"].append(gts_all)
    print(f"Completed hidden_layer_size = {size}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
