import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, feat, label):
        self.X = torch.from_numpy(feat).float()
        self.y = torch.from_numpy(label).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i]}


# Activation factory
def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "linear":
        return nn.Identity()
    raise ValueError(f"Unknown activation {name}")


# Model definitions parametrized by activation
class AIModel(nn.Module):
    def __init__(self, inp, hid, out, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid),
            get_activation(act),
            nn.Linear(hid, out),
        )

    def forward(self, x):
        return self.net(x)


class UserModel(nn.Module):
    def __init__(self, inp, hid, out, act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid),
            get_activation(act),
            nn.Linear(hid, out),
        )

    def forward(self, x):
        return self.net(x)


# Ablation over activation functions
activation_variants = ["relu", "tanh", "leaky_relu", "linear"]
experiment_data = {}

# Fixed hyperparameters
ai_bs, usr_bs = 32, 32
ai_epochs, usr_epochs = 15, 20
lr = 1e-2

for act in activation_variants:
    data_dict = {
        "ai_losses": {"train": [], "val": []},
        "ai_metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # AI data loaders
    ai_tr_loader = DataLoader(
        SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True
    )
    ai_val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=ai_bs)
    # Initialize AI model
    ai_model = AIModel(D, 16, 2, act).to(device)
    criterion_ai = nn.CrossEntropyLoss()
    opt_ai = optim.Adam(ai_model.parameters(), lr=lr)
    # Train AI
    for _ in range(ai_epochs):
        ai_model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for b in ai_tr_loader:
            x, lbl = b["x"].to(device), b["y"].to(device)
            out = ai_model(x)
            loss = criterion_ai(out, lbl)
            opt_ai.zero_grad()
            loss.backward()
            opt_ai.step()
            t_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            t_corr += (preds == lbl).sum().item()
            t_tot += x.size(0)
        data_dict["ai_losses"]["train"].append(t_loss / t_tot)
        data_dict["ai_metrics"]["train"].append(t_corr / t_tot)
        ai_model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for b in ai_val_loader:
                x, lbl = b["x"].to(device), b["y"].to(device)
                out = ai_model(x)
                loss = criterion_ai(out, lbl)
                v_loss += loss.item() * x.size(0)
                preds = out.argmax(1)
                v_corr += (preds == lbl).sum().item()
                v_tot += x.size(0)
        data_dict["ai_losses"]["val"].append(v_loss / v_tot)
        data_dict["ai_metrics"]["val"].append(v_corr / v_tot)
    # Generate soft labels
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
    # User data loaders
    X_usr_train = np.hstack([X_train, p_train])
    X_usr_val = np.hstack([X_val, p_val])
    X_usr_test = np.hstack([X_test, p_test])
    usr_tr_loader = DataLoader(
        UserDS(X_usr_train, f_train), batch_size=usr_bs, shuffle=True
    )
    usr_val_loader = DataLoader(UserDS(X_usr_val, f_val), batch_size=usr_bs)
    usr_test_loader = DataLoader(UserDS(X_usr_test, f_test), batch_size=usr_bs)
    # Initialize User model
    user_model = UserModel(D + 2, 8, 2, act).to(device)
    criterion_usr = nn.CrossEntropyLoss()
    opt_usr = optim.Adam(user_model.parameters(), lr=lr)
    # Train User
    for _ in range(usr_epochs):
        user_model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for b in usr_tr_loader:
            feat, lbl = b["feat"].to(device), b["label"].to(device)
            out = user_model(feat)
            loss = criterion_usr(out, lbl)
            opt_usr.zero_grad()
            loss.backward()
            opt_usr.step()
            t_loss += loss.item() * feat.size(0)
            preds = out.argmax(1)
            t_corr += (preds == lbl).sum().item()
            t_tot += feat.size(0)
        data_dict["losses"]["train"].append(t_loss / t_tot)
        data_dict["metrics"]["train"].append(t_corr / t_tot)
        user_model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for b in usr_val_loader:
                feat, lbl = b["feat"].to(device), b["label"].to(device)
                out = user_model(feat)
                loss = criterion_usr(out, lbl)
                v_loss += loss.item() * feat.size(0)
                preds = out.argmax(1)
                v_corr += (preds == lbl).sum().item()
                v_tot += feat.size(0)
        data_dict["losses"]["val"].append(v_loss / v_tot)
        data_dict["metrics"]["val"].append(v_corr / v_tot)
    # Test evaluation
    test_preds, test_gt = [], []
    user_model.eval()
    with torch.no_grad():
        for b in usr_test_loader:
            feat, lbl = b["feat"].to(device), b["label"].to(device)
            out = user_model(feat)
            test_preds.extend(out.argmax(1).cpu().numpy().tolist())
            test_gt.extend(lbl.cpu().numpy().tolist())
    data_dict["predictions"] = np.array(test_preds)
    data_dict["ground_truth"] = np.array(test_gt)
    # Convert lists to numpy arrays
    for k in ["ai_losses", "ai_metrics", "losses", "metrics"]:
        for phase in ["train", "val"]:
            data_dict[k][phase] = np.array(data_dict[k][phase])
    experiment_data[act] = {"synthetic": data_dict}

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
