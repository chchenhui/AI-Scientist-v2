import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Synthetic dataset generation
N, D = 2000, 2
X = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)
idx = np.random.permutation(N)
train_idx, val_idx, test_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]
mean, std = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# Dataset classes
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


# Model builder supporting variable depth
def build_model(inp_dim, hid_dim, out_dim, depth):
    if depth == 0:
        return nn.Linear(inp_dim, out_dim)
    layers = []
    if depth >= 1:
        layers += [nn.Linear(inp_dim, hid_dim), nn.ReLU()]
    if depth == 2:
        layers += [nn.Linear(hid_dim, hid_dim), nn.ReLU()]
    layers += [nn.Linear(hid_dim, out_dim)]
    return nn.Sequential(*layers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed hyperparameters
ai_bs, usr_bs = 32, 32
ai_hid, usr_hid = 16, 8
depths = [0, 1, 2]

# Prepare experiment data container
experiment_data = {"network_depth_ablation": {"synthetic": {}}}

# Pre-build AI data loaders
ai_tr_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)

for ai_depth in depths:
    # Train AI model
    ai_model = build_model(D, ai_hid, 2, ai_depth).to(device)
    optim_ai = optim.Adam(ai_model.parameters(), lr=1e-2)
    crit_ai = nn.CrossEntropyLoss()
    for _ in range(15):
        ai_model.train()
        for b in ai_tr_loader:
            x, yb = b["x"].to(device), b["y"].to(device)
            out = ai_model(x)
            loss = crit_ai(out, yb)
            optim_ai.zero_grad()
            loss.backward()
            optim_ai.step()
    # Generate pseudo-labels
    ai_model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
        logits_all = ai_model(X_all)
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
    p_train = probs_all[: len(X_train)]
    p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
    p_test = probs_all[-len(X_test) :]
    f_train, f_val, f_test = p_train.argmax(1), p_val.argmax(1), p_test.argmax(1)
    # Prepare User datasets/loaders
    X_utrain = np.hstack([X_train, p_train])
    X_uval = np.hstack([X_val, p_val])
    X_utest = np.hstack([X_test, p_test])
    usr_tr_loader = DataLoader(
        UserDS(X_utrain, f_train), batch_size=usr_bs, shuffle=True
    )
    usr_val_loader = DataLoader(UserDS(X_uval, f_val), batch_size=usr_bs)
    usr_test_loader = DataLoader(UserDS(X_utest, f_test), batch_size=usr_bs)
    # Loop over user depths
    for usr_depth in depths:
        user_model = build_model(D + 2, usr_hid, 2, usr_depth).to(device)
        optim_usr = optim.Adam(user_model.parameters(), lr=1e-2)
        crit_usr = nn.CrossEntropyLoss()
        train_accs, val_accs = [], []
        train_losses, val_losses = [], []
        for _ in range(20):
            # train
            user_model.train()
            tloss, correct, total = 0.0, 0, 0
            for b in usr_tr_loader:
                feat, lbl = b["feat"].to(device), b["label"].to(device)
                out = user_model(feat)
                loss = crit_usr(out, lbl)
                optim_usr.zero_grad()
                loss.backward()
                optim_usr.step()
                tloss += loss.item() * feat.size(0)
                preds = out.argmax(1)
                correct += (preds == lbl).sum().item()
                total += lbl.size(0)
            train_losses.append(tloss / total)
            train_accs.append(correct / total)
            # val
            user_model.eval()
            vloss, vcorr, vtot = 0.0, 0, 0
            with torch.no_grad():
                for b in usr_val_loader:
                    feat, lbl = b["feat"].to(device), b["label"].to(device)
                    out = user_model(feat)
                    loss = crit_usr(out, lbl)
                    vloss += loss.item() * feat.size(0)
                    preds = out.argmax(1)
                    vcorr += (preds == lbl).sum().item()
                    vtot += lbl.size(0)
            val_losses.append(vloss / vtot)
            val_accs.append(vcorr / vtot)
        # test
        test_preds, test_gt = [], []
        user_model.eval()
        with torch.no_grad():
            for b in usr_test_loader:
                feat, lbl = b["feat"].to(device), b["label"].to(device)
                out = user_model(feat)
                pred = out.argmax(1).cpu().numpy().tolist()
                test_preds.extend(pred)
                test_gt.extend(lbl.cpu().numpy().tolist())
        key = f"ai_depth_{ai_depth}_usr_depth_{usr_depth}"
        experiment_data["network_depth_ablation"]["synthetic"][key] = {
            "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
            "losses": {"train": np.array(train_losses), "val": np.array(val_losses)},
            "predictions": np.array(test_preds),
            "ground_truth": np.array(test_gt),
        }

# Save data
np.save("experiment_data.npy", experiment_data)
