import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# GPU/CPU setup
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


# Model
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


model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Experiment logging
experiment_data = {
    "synthetic_xor": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Training + evaluation
epochs = 10
mc_T = 5
threshold = 0.02
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
    experiment_data["synthetic_xor"]["losses"]["train"].append(train_loss)

    # Compute CES on train
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
                # MC-dropout
                model.train()
                ps = []
                for _ in range(mc_T):
                    p = torch.softmax(model(xi), dim=1)
                    ps.append(p.cpu().numpy())
                ps = np.stack(ps, 0)
                var = ps.var(0).sum()
                model.eval()
                if var > threshold:
                    clar_count += 1
                    out_c = model(Xb[i : i + 1])
                    clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                else:
                    clar_corr += (preds_base[i] == yb[i]).item()
    base_acc_tr = base_corr / len(train_ds)
    clar_acc_tr = clar_corr / len(train_ds)
    avg_ct_tr = clar_count / len(train_ds) if len(train_ds) else 0
    CES_tr = (clar_acc_tr - base_acc_tr) / avg_ct_tr if avg_ct_tr > 0 else 0.0
    experiment_data["synthetic_xor"]["metrics"]["train"].append(CES_tr)

    # Validation loss
    model.eval()
    val_loss, _ = 0.0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            val_loss += criterion(out, yb).item() * Xb.size(0)
    val_loss /= len(val_ds)
    experiment_data["synthetic_xor"]["losses"]["val"].append(val_loss)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

    # Compute CES on val
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
                var = ps.var(0).sum()
                model.eval()
                if var > threshold:
                    clar_count += 1
                    out_c = model(Xb[i : i + 1])
                    clar_corr += (out_c.argmax(1) == yb[i : i + 1]).sum().item()
                else:
                    clar_corr += (preds_base[i] == yb[i]).item()
    base_acc_val = base_corr / len(val_ds)
    clar_acc_val = clar_corr / len(val_ds)
    avg_ct_val = clar_count / len(val_ds) if len(val_ds) else 0
    CES_val = (clar_acc_val - base_acc_val) / avg_ct_val if avg_ct_val > 0 else 0.0
    experiment_data["synthetic_xor"]["metrics"]["val"].append(CES_val)

    # Save predictions and ground truth for this epoch
    preds_list, gts_list = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            preds_list.append(out.argmax(1).cpu().numpy())
            gts_list.append(yb.cpu().numpy())
    experiment_data["synthetic_xor"]["predictions"].append(np.concatenate(preds_list))
    experiment_data["synthetic_xor"]["ground_truth"].append(np.concatenate(gts_list))

    print(f"Epoch {epoch}: train_CES = {CES_tr:.4f}, val_CES = {CES_val:.4f}")

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
