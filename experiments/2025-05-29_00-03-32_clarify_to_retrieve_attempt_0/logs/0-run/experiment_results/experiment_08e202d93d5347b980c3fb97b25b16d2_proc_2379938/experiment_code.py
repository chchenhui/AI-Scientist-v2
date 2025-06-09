import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
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


# Model definition
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# Hyperparameters
epochs = 10
mc_T = 5
thresholds = [0.005, 0.01, 0.02, 0.05, 0.1]

# Prepare experiment_data structure
experiment_data = {
    "hyperparam_tuning_type_1": {
        "synthetic_xor": {
            "thresholds": thresholds,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Sweep thresholds
for thr in thresholds:
    print(f"\n=== Threshold = {thr} ===")
    # per-threshold containers
    met_tr, met_val = [], []
    los_tr, los_val = [], []
    preds_epochs, gts_epochs = [], []

    # new model & optimizer each threshold
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        # Training
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
        los_tr.append(train_loss)

        # Train CES
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
                    var = np.stack(ps, 0).var(0).sum()
                    model.eval()
                    if var > thr:
                        clar_count += 1
                        out_c = model(Xb[i : i + 1])
                        clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                    else:
                        clar_corr += int(preds_base[i] == yb[i])
        base_acc = base_corr / len(train_ds)
        clar_acc = clar_corr / len(train_ds)
        avg_ct = clar_count / len(train_ds) if len(train_ds) > 0 else 0
        CES_tr = (clar_acc - base_acc) / avg_ct if avg_ct > 0 else 0.0
        met_tr.append(CES_tr)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                val_loss += criterion(out, yb).item() * Xb.size(0)
        val_loss /= len(val_ds)
        los_val.append(val_loss)

        # Val CES
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
                    var = np.stack(ps, 0).var(0).sum()
                    model.eval()
                    if var > thr:
                        clar_count += 1
                        out_c = model(Xb[i : i + 1])
                        clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                    else:
                        clar_corr += int(preds_base[i] == yb[i])
        base_acc = base_corr / len(val_ds)
        clar_acc = clar_corr / len(val_ds)
        avg_ct = clar_count / len(val_ds) if len(val_ds) > 0 else 0
        CES_val = (clar_acc - base_acc) / avg_ct if avg_ct > 0 else 0.0
        met_val.append(CES_val)

        # record predictions & gts
        preds_list, gts_list = [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                out = model(Xb)
                preds_list.append(out.argmax(1).cpu().numpy())
                gts_list.append(yb.numpy())
        preds_epochs.append(np.concatenate(preds_list))
        gts_epochs.append(np.concatenate(gts_list))

        print(
            f"Thr {thr:.3f} Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_CES={CES_tr:.4f}, val_CES={CES_val:.4f}"
        )

    # store per-threshold results
    d = experiment_data["hyperparam_tuning_type_1"]["synthetic_xor"]
    d["metrics"]["train"].append(met_tr)
    d["metrics"]["val"].append(met_val)
    d["losses"]["train"].append(los_tr)
    d["losses"]["val"].append(los_val)
    d["predictions"].append(preds_epochs)
    d["ground_truth"].append(gts_epochs)

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
