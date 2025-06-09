import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, feat, label, conf):
        self.X = torch.from_numpy(feat).float()
        self.y = torch.from_numpy(label).long()
        self.w = torch.from_numpy(conf).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i], "weight": self.w[i]}


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ai_batch_sizes = [16, 32, 64]
usr_batch_sizes = [16, 32, 64]
threshold = 0.8

experiment_data = {"uniform": {}, "thresholded": {}, "confidence_weighted": {}}

for ai_bs in ai_batch_sizes:
    # Train AI model
    ai_tr = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)
    ai_val = DataLoader(SimpleDS(X_val, y_val), batch_size=ai_bs)
    ai = AIModel(D, 16, 2).to(device)
    opt_ai = optim.Adam(ai.parameters(), lr=1e-2)
    crit_ai = nn.CrossEntropyLoss()
    for _ in range(15):
        ai.train()
        for b in ai_tr:
            x, yb = b["x"].to(device), b["y"].to(device)
            out = ai(x)
            loss = crit_ai(out, yb)
            opt_ai.zero_grad()
            loss.backward()
            opt_ai.step()
    # Teacher preds & confidences
    ai.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
        lg = ai(X_all)
        pp = F.softmax(lg, dim=1).cpu().numpy()
    p_train = pp[: len(X_train)]
    p_val = pp[len(X_train) : len(X_train) + len(X_val)]
    p_test = pp[-len(X_test) :]
    f_train = p_train.argmax(1)
    f_val = p_val.argmax(1)
    f_test = p_test.argmax(1)
    c_train = p_train.max(1)
    c_val = p_val.max(1)
    c_test = p_test.max(1)
    # Student features include teacher probs
    X_usr_train = np.hstack([X_train, p_train])
    X_usr_val = np.hstack([X_val, p_val])
    X_usr_test = np.hstack([X_test, p_test])

    for ablation in ["uniform", "thresholded", "confidence_weighted"]:
        for usr_bs in usr_batch_sizes:
            tr_ds = UserDS(X_usr_train, f_train, c_train)
            val_ds = UserDS(X_usr_val, f_val, c_val)
            te_ds = UserDS(X_usr_test, f_test, c_test)
            tr_ld = DataLoader(tr_ds, batch_size=usr_bs, shuffle=True)
            val_ld = DataLoader(val_ds, batch_size=usr_bs)
            te_ld = DataLoader(te_ds, batch_size=usr_bs)
            user = UserModel(D + 2, 8, 2).to(device)
            opt_u = optim.Adam(user.parameters(), lr=1e-2)

            train_accs, val_accs = [], []
            train_losses, val_losses = [], []

            for _ in range(20):
                # train
                user.train()
                num_sum = 0.0
                w_sum = 0.0
                corr = tot = 0
                for b in tr_ld:
                    feat = b["feat"].to(device)
                    lbl = b["label"].to(device)
                    wconf = b["weight"].to(device)
                    # select weights
                    if ablation == "uniform":
                        w = torch.ones_like(wconf)
                    elif ablation == "thresholded":
                        w = (wconf >= threshold).float()
                    else:
                        w = wconf
                    out = user(feat)
                    losses = F.cross_entropy(out, lbl, reduction="none")
                    weighted = (losses * w).sum() / (w.sum() + 1e-6)
                    opt_u.zero_grad()
                    weighted.backward()
                    opt_u.step()
                    num_sum += (losses * w).sum().item()
                    w_sum += w.sum().item()
                    preds = out.argmax(1)
                    corr += (preds == lbl).sum().item()
                    tot += lbl.size(0)
                train_losses.append(num_sum / (w_sum + 1e-6))
                train_accs.append(corr / tot)

                # val
                user.eval()
                num_sum = 0.0
                w_sum = 0.0
                v_corr = v_tot = 0
                with torch.no_grad():
                    for b in val_ld:
                        feat = b["feat"].to(device)
                        lbl = b["label"].to(device)
                        wconf = b["weight"].to(device)
                        if ablation == "uniform":
                            w = torch.ones_like(wconf)
                        elif ablation == "thresholded":
                            w = (wconf >= threshold).float()
                        else:
                            w = wconf
                        out = user(feat)
                        losses = F.cross_entropy(out, lbl, reduction="none")
                        num_sum += (losses * w).sum().item()
                        w_sum += w.sum().item()
                        preds = out.argmax(1)
                        v_corr += (preds == lbl).sum().item()
                        v_tot += lbl.size(0)
                val_losses.append(num_sum / (w_sum + 1e-6))
                val_accs.append(v_corr / v_tot)

            # test
            test_preds = []
            test_gt = []
            user.eval()
            with torch.no_grad():
                for b in te_ld:
                    feat = b["feat"].to(device)
                    lbl = b["label"].to(device)
                    out = user(feat)
                    p = out.argmax(1).cpu().numpy().tolist()
                    test_preds.extend(p)
                    test_gt.extend(lbl.cpu().numpy().tolist())

            key = f"ai_bs_{ai_bs}_user_bs_{usr_bs}"
            experiment_data[ablation][key] = {
                "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
                "losses": {
                    "train": np.array(train_losses),
                    "val": np.array(val_losses),
                },
                "predictions": np.array(test_preds),
                "ground_truth": np.array(test_gt),
            }

# Save
np.save("experiment_data.npy", experiment_data)
