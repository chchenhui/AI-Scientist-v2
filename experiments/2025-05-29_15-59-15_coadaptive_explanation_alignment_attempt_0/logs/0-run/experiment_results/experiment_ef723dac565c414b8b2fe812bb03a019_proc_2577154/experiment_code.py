import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device & seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)

# full synthetic data
N, D = 2000, 2
X_full = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X_full.dot(w_true) + b_true
probs_full = 1 / (1 + np.exp(-logits))
y_full = (np.random.rand(N) < probs_full).astype(int)


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


# model defs
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


# hyperparameters
ai_batch_sizes = [16, 32, 64]
usr_batch_sizes = [16, 32, 64]

# prepare experiment container
experiment_data = {"class_imbalance": {}}
ratios = [0.5, 0.7, 0.9]
names = ["50_50", "70_30", "90_10"]

for ratio, name in zip(ratios, names):
    # compute counts for class0 (majority) and class1
    n0 = int(N * ratio)
    n1 = N - n0
    idx0 = np.where(y_full == 0)[0]
    idx1 = np.where(y_full == 1)[0]
    # sample with/without replacement
    sel0 = np.random.choice(idx0, n0, replace=(len(idx0) < n0))
    sel1 = np.random.choice(idx1, n1, replace=(len(idx1) < n1))
    idxs = np.concatenate([sel0, sel1])
    np.random.shuffle(idxs)
    X, y = X_full[idxs], y_full[idxs]
    # split
    tr_i, val_i, te_i = np.arange(0, 1200), np.arange(1200, 1500), np.arange(1500, N)
    X_train, y_train = X[tr_i], y[tr_i]
    X_val, y_val = X[val_i], y[val_i]
    X_test, y_test = X[te_i], y[te_i]
    # normalize
    mean, std = X_train.mean(0), X_train.std(0) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    experiment_data["class_imbalance"][name] = {}

    for ai_bs in ai_batch_sizes:
        # AI loaders
        ai_tr = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)
        ai_val = DataLoader(SimpleDS(X_val, y_val), batch_size=ai_bs)
        # AI model
        ai_model = AIModel(D, 16, 2).to(device)
        crit_ai = nn.CrossEntropyLoss()
        opt_ai = optim.Adam(ai_model.parameters(), lr=1e-2)
        # train AI
        for _ in range(15):
            ai_model.train()
            for b in ai_tr:
                x = b["x"].to(device)
                yb = b["y"].to(device)
                out = ai_model(x)
                loss = crit_ai(out, yb)
                opt_ai.zero_grad()
                loss.backward()
                opt_ai.step()
        # get probs & preds
        ai_model.eval()
        with torch.no_grad():
            X_all = (
                torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
            )
            logits_all = ai_model(X_all)
            probs_all = torch.softmax(logits_all, 1).cpu().numpy()
        p_tr = probs_all[: len(X_train)]
        p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
        p_te = probs_all[-len(X_test) :]
        f_tr, f_val, f_te = p_tr.argmax(1), p_val.argmax(1), p_te.argmax(1)
        # user features
        X_usr_tr = np.hstack([X_train, p_tr])
        X_usr_val = np.hstack([X_val, p_val])
        X_usr_te = np.hstack([X_test, p_te])

        for usr_bs in usr_batch_sizes:
            usr_tr = DataLoader(UserDS(X_usr_tr, f_tr), batch_size=usr_bs, shuffle=True)
            usr_val = DataLoader(UserDS(X_usr_val, f_val), batch_size=usr_bs)
            usr_te = DataLoader(UserDS(X_usr_te, f_te), batch_size=usr_bs)
            user_model = UserModel(D + 2, 8, 2).to(device)
            crit_u = nn.CrossEntropyLoss()
            opt_u = optim.Adam(user_model.parameters(), lr=1e-2)
            train_accs, val_accs = [], []
            train_losses, val_losses = [], []

            # train user
            for _ in range(20):
                user_model.train()
                t_loss, corr, tot = 0, 0, 0
                for b in usr_tr:
                    feat = b["feat"].to(device)
                    lbl = b["label"].to(device)
                    out = user_model(feat)
                    loss = crit_u(out, lbl)
                    opt_u.zero_grad()
                    loss.backward()
                    opt_u.step()
                    t_loss += loss.item() * feat.size(0)
                    preds = out.argmax(1)
                    corr += (preds == lbl).sum().item()
                    tot += lbl.size(0)
                train_losses.append(t_loss / tot)
                train_accs.append(corr / tot)
                user_model.eval()
                v_loss, v_corr, v_tot = 0, 0, 0
                with torch.no_grad():
                    for b in usr_val:
                        feat = b["feat"].to(device)
                        lbl = b["label"].to(device)
                        out = user_model(feat)
                        loss = crit_u(out, lbl)
                        v_loss += loss.item() * feat.size(0)
                        p = out.argmax(1)
                        v_corr += (p == lbl).sum().item()
                        v_tot += lbl.size(0)
                val_losses.append(v_loss / v_tot)
                val_accs.append(v_corr / v_tot)

            # test user
            test_preds, test_gt = [], []
            user_model.eval()
            with torch.no_grad():
                for b in usr_te:
                    feat = b["feat"].to(device)
                    lbl = b["label"].to(device)
                    out = user_model(feat)
                    test_preds.extend(out.argmax(1).cpu().numpy().tolist())
                    test_gt.extend(lbl.cpu().numpy().tolist())

            key = f"ai_bs_{ai_bs}_user_bs_{usr_bs}"
            experiment_data["class_imbalance"][name][key] = {
                "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
                "losses": {
                    "train": np.array(train_losses),
                    "val": np.array(val_losses),
                },
                "predictions": np.array(test_preds),
                "ground_truth": np.array(test_gt),
            }

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
