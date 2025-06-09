import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# set up working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)

# generate synthetic data
N, D = 2000, 2
X = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# train/val/test split
idx = np.random.permutation(N)
train_idx, val_idx, test_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# normalize features
mean, std = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# dataset classes
class SimpleDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


class UserDS(Dataset):
    def __init__(self, feat, label):
        self.X = torch.from_numpy(feat).float()
        self.y = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i]}


# model definitions
class AIModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class UserModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# train AI model once
ai_bs = 32
ai_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)
ai_model = AIModel(D, 16, 2).to(device)
opt_ai = optim.Adam(ai_model.parameters(), lr=1e-2)
crit_ai = nn.CrossEntropyLoss()
for _ in range(15):
    ai_model.train()
    for b in ai_loader:
        x, lbl = b["x"].to(device), b["y"].to(device)
        out = ai_model(x)
        loss = crit_ai(out, lbl)
        opt_ai.zero_grad()
        loss.backward()
        opt_ai.step()

# get raw logits
ai_model.eval()
with torch.no_grad():
    X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
    logits_all = ai_model(X_all)

# ablation: temperature scaling
temperatures = [0.5, 1.0, 2.0, 5.0]
usr_bs = 32
experiment_data = {"temperature_scaling": {}}

for T in temperatures:
    # scale logits and compute probs
    scaled = logits_all / T
    probs_all = torch.softmax(scaled, dim=1).cpu().numpy()
    p_train = probs_all[: len(X_train)]
    p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
    p_test = probs_all[-len(X_test) :]

    # pseudo-labels from argmax
    f_train, f_val, f_test = p_train.argmax(1), p_val.argmax(1), p_test.argmax(1)

    # construct user features
    X_utrain = np.hstack([X_train, p_train])
    X_uval = np.hstack([X_val, p_val])
    X_utest = np.hstack([X_test, p_test])

    # data loaders for user
    tr_loader = DataLoader(UserDS(X_utrain, f_train), batch_size=usr_bs, shuffle=True)
    va_loader = DataLoader(UserDS(X_uval, f_val), batch_size=usr_bs)
    te_loader = DataLoader(UserDS(X_utest, f_test), batch_size=usr_bs)

    # train user model
    user_model = UserModel(D + 2, 8, 2).to(device)
    opt_usr = optim.Adam(user_model.parameters(), lr=1e-2)
    crit_usr = nn.CrossEntropyLoss()

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for _ in range(20):
        user_model.train()
        tloss = 0
        tcor = 0
        ttot = 0
        for b in tr_loader:
            x, lbl = b["feat"].to(device), b["label"].to(device)
            out = user_model(x)
            loss = crit_usr(out, lbl)
            opt_usr.zero_grad()
            loss.backward()
            opt_usr.step()
            tloss += loss.item() * x.size(0)
            tcor += (out.argmax(1) == lbl).sum().item()
            ttot += lbl.size(0)
        train_losses.append(tloss / ttot)
        train_accs.append(tcor / ttot)

        user_model.eval()
        vloss = 0
        vcor = 0
        vtot = 0
        with torch.no_grad():
            for b in va_loader:
                x, lbl = b["feat"].to(device), b["label"].to(device)
                out = user_model(x)
                loss = crit_usr(out, lbl)
                vloss += loss.item() * x.size(0)
                vcor += (out.argmax(1) == lbl).sum().item()
                vtot += lbl.size(0)
        val_losses.append(vloss / vtot)
        val_accs.append(vcor / vtot)

    # test evaluation
    test_preds, test_gt = [], []
    user_model.eval()
    with torch.no_grad():
        for b in te_loader:
            x, lbl = b["feat"].to(device), b["label"].to(device)
            preds = user_model(x).argmax(1).cpu().numpy()
            test_preds.extend(preds.tolist())
            test_gt.extend(lbl.cpu().numpy().tolist())

    # record results
    key = f"T_{T}"
    experiment_data["temperature_scaling"][key] = {
        "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
        "losses": {"train": np.array(train_losses), "val": np.array(val_losses)},
        "predictions": np.array(test_preds),
        "ground_truth": np.array(test_gt),
    }

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
