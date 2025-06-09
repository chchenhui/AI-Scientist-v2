import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Reproducibility and device
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic dataset
N, D = 2000, 2
X = np.random.randn(N, D)
w_true, b_true = np.array([2.0, -3.0]), 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# Split and normalize
idx = np.random.permutation(N)
tr, va, te = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[tr], y[tr]
X_val, y_val = X[va], y[va]
X_test, y_test = X[te], y[te]
mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - mu) / sd
X_val = (X_val - mu) / sd
X_test = (X_test - mu) / sd


# Datasets
class SimpleDS(Dataset):
    def __init__(self, X, y):
        self.x, self.y = torch.from_numpy(X).float(), torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class UserDS(Dataset):
    def __init__(self, feat, label):
        self.f, self.y = torch.from_numpy(feat).float(), torch.from_numpy(label).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.f[i], "label": self.y[i]}


# Models
class AIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


class UserModel(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


# Ablation study: original-feature removal
ai_bs_list = [16, 32, 64]
usr_bs_list = [16, 32, 64]
experiment_data = {"original_feature_removal": {"synthetic": {}}}

for ai_bs in ai_bs_list:
    ai_tr = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)
    ai_val = DataLoader(SimpleDS(X_val, y_val), batch_size=ai_bs)
    model_ai = AIModel().to(device)
    opt_ai = optim.Adam(model_ai.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    for _ in range(15):
        model_ai.train()
        for b in ai_tr:
            x, yb = b["x"].to(device), b["y"].to(device)
            loss = crit(model_ai(x), yb)
            opt_ai.zero_grad()
            loss.backward()
            opt_ai.step()
    # get softmax outputs
    model_ai.eval()
    with torch.no_grad():
        Xall = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
        ps = torch.softmax(model_ai(Xall), 1).cpu().numpy()
    p_tr, p_va, p_te = (
        ps[: len(X_train)],
        ps[len(X_train) : len(X_train) + len(X_val)],
        ps[-len(X_test) :],
    )
    f_tr, f_va, f_te = p_tr.argmax(1), p_va.argmax(1), p_te.argmax(1)

    # only probabilities as features
    for usr_bs in usr_bs_list:
        usr_tr = DataLoader(UserDS(p_tr, f_tr), batch_size=usr_bs, shuffle=True)
        usr_val = DataLoader(UserDS(p_va, f_va), batch_size=usr_bs)
        usr_test = DataLoader(UserDS(p_te, f_te), batch_size=usr_bs)
        user = UserModel(inp=2).to(device)
        opt_u = optim.Adam(user.parameters(), lr=1e-2)
        acc_tr, acc_va, ls_tr, ls_va = [], [], [], []
        for _ in range(20):
            user.train()
            tot, corr, L = 0, 0, 0.0
            for b in usr_tr:
                f, lbl = b["feat"].to(device), b["label"].to(device)
                out = user(f)
                loss = crit(out, lbl)
                opt_u.zero_grad()
                loss.backward()
                opt_u.step()
                L += loss.item() * f.size(0)
                preds = out.argmax(1)
                corr += (preds == lbl).sum().item()
                tot += lbl.size(0)
            ls_tr.append(L / tot)
            acc_tr.append(corr / tot)
            user.eval()
            with torch.no_grad():
                vt, vc, vL = 0, 0, 0.0
                for b in usr_val:
                    f, l = b["feat"].to(device), b["label"].to(device)
                    o = user(f)
                    loss = crit(o, l)
                    vL += loss.item() * f.size(0)
                    p = o.argmax(1)
                    vc += (p == l).sum().item()
                    vt += l.size(0)
                ls_va.append(vL / vt)
                acc_va.append(vc / vt)
        # test
        preds, gts = [], []
        user.eval()
        with torch.no_grad():
            for b in usr_test:
                f, l = b["feat"].to(device), b["label"].to(device)
                p = user(f).argmax(1).cpu().numpy()
                preds.extend(p.tolist())
                gts.extend(l.cpu().numpy().tolist())
        key = f"ai_bs_{ai_bs}_usr_bs_{usr_bs}"
        experiment_data["original_feature_removal"]["synthetic"][key] = {
            "metrics": {"train": np.array(acc_tr), "val": np.array(acc_va)},
            "losses": {"train": np.array(ls_tr), "val": np.array(ls_va)},
            "predictions": np.array(preds),
            "ground_truth": np.array(gts),
        }

np.save("experiment_data.npy", experiment_data)
