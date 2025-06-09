import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Synthetic data
N, D = 2000, 2
X = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# Split
idx = np.random.permutation(N)
train_idx, val_idx, test_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Normalize
mean, std = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# Datasets
class SimpleDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


class UserDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i]}


# Models
class AIModel(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, x):
        return self.net(x)


class UserModel(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, x):
        return self.net(x)


# Train teacher
ai_bs = 32
ai_tr_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)
ai_val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=ai_bs, shuffle=False)
ai_model = AIModel(D, 16, 2).to(device)
crit_ai = nn.CrossEntropyLoss()
opt_ai = optim.Adam(ai_model.parameters(), lr=1e-2)
for _ in range(15):
    ai_model.train()
    for batch in ai_tr_loader:
        x, yb = batch["x"].to(device), batch["y"].to(device)
        out = ai_model(x)
        loss = crit_ai(out, yb)
        opt_ai.zero_grad()
        loss.backward()
        opt_ai.step()

# Get teacher probs & pseudo-labels
ai_model.eval()
with torch.no_grad():
    X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
    logits_all = ai_model(X_all)
    probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
p_train = probs_all[: len(X_train)]
p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
p_test = probs_all[-len(X_test) :]
f_train = p_train.argmax(1)
f_val = p_val.argmax(1)
f_test = p_test.argmax(1)

# Build user features
X_usr_train = np.hstack([X_train, p_train])
X_usr_val = np.hstack([X_val, p_val])
X_usr_test = np.hstack([X_test, p_test])

# Ablation: confidence thresholds
thresholds = [0.6, 0.8, 0.9]
experiment_data = {"confidence_filter": {}}
usr_bs = 32

for thr in thresholds:
    # filter train
    keep = np.where(np.max(p_train, axis=1) >= thr)[0]
    X_tr_f = X_usr_train[keep]
    y_tr_f = f_train[keep]
    # loaders
    usr_tr = DataLoader(UserDS(X_tr_f, y_tr_f), batch_size=usr_bs, shuffle=True)
    usr_val = DataLoader(UserDS(X_usr_val, f_val), batch_size=usr_bs, shuffle=False)
    usr_test = DataLoader(UserDS(X_usr_test, f_test), batch_size=usr_bs, shuffle=False)
    # user model
    user = UserModel(D + 2, 8, 2).to(device)
    crit_u = nn.CrossEntropyLoss()
    opt_u = optim.Adam(user.parameters(), lr=1e-2)
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    # train
    for _ in range(20):
        user.train()
        t_loss, corr, tot = 0.0, 0, 0
        for b in usr_tr:
            x = b["feat"].to(device)
            yb = b["label"].to(device)
            out = user(x)
            loss = crit_u(out, yb)
            opt_u.zero_grad()
            loss.backward()
            opt_u.step()
            t_loss += loss.item() * len(yb)
            pred = out.argmax(1)
            corr += (pred == yb).sum().item()
            tot += len(yb)
        train_losses.append(t_loss / tot)
        train_accs.append(corr / tot)
        # val
        user.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for b in usr_val:
                x = b["feat"].to(device)
                yb = b["label"].to(device)
                out = user(x)
                loss = crit_u(out, yb)
                v_loss += loss.item() * len(yb)
                pred = out.argmax(1)
                v_corr += (pred == yb).sum().item()
                v_tot += len(yb)
        val_losses.append(v_loss / v_tot)
        val_accs.append(v_corr / v_tot)
    # test
    user.eval()
    test_preds, test_gt = [], []
    with torch.no_grad():
        for b in usr_test:
            x = b["feat"].to(device)
            yb = b["label"].to(device)
            out = user(x)
            test_preds.extend(out.argmax(1).cpu().numpy().tolist())
            test_gt.extend(yb.cpu().numpy().tolist())
    # save
    key = f"threshold_{thr}"
    experiment_data["confidence_filter"][key] = {
        "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
        "losses": {"train": np.array(train_losses), "val": np.array(val_losses)},
        "predictions": np.array(test_preds),
        "ground_truth": np.array(test_gt),
    }

# dump
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
