import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic dataset generation
np.random.seed(0)
N, D = 2000, 2
X = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# Train/val/test split
idx = np.random.permutation(N)
train_idx, val_idx, test_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Normalize features
mean, std = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# AI model definition & training
class AIModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


ai_model = AIModel(D, 16, 2).to(device)
criterion_ai = nn.CrossEntropyLoss()
optimizer_ai = optim.Adam(ai_model.parameters(), lr=1e-2)


class SimpleDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


ai_train_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=32, shuffle=True)
ai_val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=32)
for epoch in range(15):
    ai_model.train()
    for batch in ai_train_loader:
        xb, yb = batch["x"].to(device), batch["y"].to(device)
        out = ai_model(xb)
        loss = criterion_ai(out, yb)
        optimizer_ai.zero_grad()
        loss.backward()
        optimizer_ai.step()
    ai_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in ai_val_loader:
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            val_loss += criterion_ai(ai_model(xb), yb).item() * xb.size(0)
    val_loss /= len(ai_val_loader.dataset)
    print(f"Epoch {epoch}: AI val loss = {val_loss:.4f}")

# Generate AI explanations
ai_model.eval()
with torch.no_grad():
    X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
    logits_all = ai_model(X_all)
    probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
p_train = probs_all[: len(X_train)]
p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
p_test = probs_all[-len(X_test) :]
f_train = np.argmax(p_train, axis=1)
f_val = np.argmax(p_val, axis=1)
f_test = np.argmax(p_test, axis=1)

# Prepare user dataset
X_usr_train = np.hstack([X_train, p_train])
X_usr_val = np.hstack([X_val, p_val])
X_usr_test = np.hstack([X_test, p_test])


class UserDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i]}


usr_train_loader = DataLoader(UserDS(X_usr_train, f_train), batch_size=32, shuffle=True)
usr_val_loader = DataLoader(UserDS(X_usr_val, f_val), batch_size=32)
usr_test_loader = DataLoader(UserDS(X_usr_test, f_test), batch_size=32)


# User model definition
class UserModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# Hyperparameter sweep on weight_decay
weight_decays = np.logspace(-5, -2, 4)
experiment_data = {
    "weight_decay": {
        "static_explainer": {
            "weight_decays": weight_decays.tolist(),
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for wd in weight_decays:
    usr_model = UserModel(D + 2, 8, 2).to(device)
    optimizer_usr = optim.Adam(usr_model.parameters(), lr=1e-2, weight_decay=wd)
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    for epoch in range(20):
        usr_model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for batch in usr_train_loader:
            xb, yb = batch["feat"].to(device), batch["label"].to(device)
            out = usr_model(xb)
            loss = nn.CrossEntropyLoss()(out, yb)
            optimizer_usr.zero_grad()
            loss.backward()
            optimizer_usr.step()
            t_loss += loss.item() * xb.size(0)
            preds = out.argmax(1)
            t_corr += (preds == yb).sum().item()
            t_tot += len(yb)
        train_losses.append(t_loss / t_tot)
        train_accs.append(t_corr / t_tot)
        usr_model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for batch in usr_val_loader:
                xb, yb = batch["feat"].to(device), batch["label"].to(device)
                out = usr_model(xb)
                loss = nn.CrossEntropyLoss()(out, yb)
                v_loss += loss.item() * xb.size(0)
                preds = out.argmax(1)
                v_corr += (preds == yb).sum().item()
                v_tot += len(yb)
        val_losses.append(v_loss / v_tot)
        val_accs.append(v_corr / v_tot)
    # store per‐WD results
    sd = experiment_data["weight_decay"]["static_explainer"]
    sd["metrics"]["train"].append(train_accs)
    sd["metrics"]["val"].append(val_accs)
    sd["losses"]["train"].append(train_losses)
    sd["losses"]["val"].append(val_losses)
    # test predictions
    usr_model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in usr_test_loader:
            xb = batch["feat"].to(device)
            test_preds.extend(usr_model(xb).argmax(1).cpu().numpy().tolist())
    sd["predictions"].append(test_preds)
    if not sd["ground_truth"]:
        sd["ground_truth"] = f_test.tolist()

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
