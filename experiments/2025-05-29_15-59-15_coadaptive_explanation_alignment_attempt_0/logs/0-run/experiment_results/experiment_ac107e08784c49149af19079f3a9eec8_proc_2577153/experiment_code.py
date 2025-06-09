import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Synthetic dataset
N, D = 2000, 2
X = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# Split & normalize
idx = np.random.permutation(N)
train_idx, val_idx, test_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]
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
    def __init__(self, feat, label):
        self.X = torch.from_numpy(feat).float()
        self.y = torch.from_numpy(label).long()

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


# Hyperparams
ai_batch_sizes = [16, 32, 64]
usr_batch_sizes = [16, 32, 64]
noise_levels = [0.0, 0.1, 0.2, 0.5]
experiment_data = {"teacher_prob_noise": {}}

for ai_bs in ai_batch_sizes:
    # Train AI model
    ai_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)
    ai_model = AIModel(D, 16, 2).to(device)
    criterion_ai = nn.CrossEntropyLoss()
    optimizer_ai = optim.Adam(ai_model.parameters(), lr=1e-2)
    for epoch in range(15):
        ai_model.train()
        for batch in ai_loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            x, yb = batch["x"], batch["y"]
            out = ai_model(x)
            loss = criterion_ai(out, yb)
            optimizer_ai.zero_grad()
            loss.backward()
            optimizer_ai.step()
    # Get teacher probs
    ai_model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
        logits_all = ai_model(X_all)
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
    p_train = probs_all[: len(X_train)]
    p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
    p_test = probs_all[-len(X_test) :]

    for sigma in noise_levels:
        skey = f"sigma_{sigma}"
        experiment_data["teacher_prob_noise"].setdefault(skey, {"batch_size": {}})

        # Inject noise + renormalize with epsilon
        noise_t = np.random.normal(0, sigma, p_train.shape)
        noise_v = np.random.normal(0, sigma, p_val.shape)
        noise_te = np.random.normal(0, sigma, p_test.shape)
        p_train_n = np.clip(p_train + noise_t, 0, 1)
        p_val_n = np.clip(p_val + noise_v, 0, 1)
        p_test_n = np.clip(p_test + noise_te, 0, 1)
        p_train_n /= p_train_n.sum(axis=1, keepdims=True) + 1e-8
        p_val_n /= p_val_n.sum(axis=1, keepdims=True) + 1e-8
        p_test_n /= p_test_n.sum(axis=1, keepdims=True) + 1e-8

        f_train_n, f_val_n, f_test_n = (
            p_train_n.argmax(1),
            p_val_n.argmax(1),
            p_test_n.argmax(1),
        )
        X_usr_train = np.hstack([X_train, p_train_n])
        X_usr_val = np.hstack([X_val, p_val_n])
        X_usr_test = np.hstack([X_test, p_test_n])

        for usr_bs in usr_batch_sizes:
            usr_tr = DataLoader(
                UserDS(X_usr_train, f_train_n), batch_size=usr_bs, shuffle=True
            )
            usr_val = DataLoader(UserDS(X_usr_val, f_val_n), batch_size=usr_bs)
            usr_te = DataLoader(UserDS(X_usr_test, f_test_n), batch_size=usr_bs)

            user_model = UserModel(D + 2, 8, 2).to(device)
            criterion_usr = nn.CrossEntropyLoss()
            optimizer_usr = optim.Adam(user_model.parameters(), lr=1e-2)

            train_accs, val_accs = [], []
            train_losses, val_losses = [], []
            alignment_scores = []

            for epoch in range(1, 21):
                # Train
                user_model.train()
                t_loss = corr = tot = 0
                for batch in usr_tr:
                    batch = {
                        k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                    }
                    feat, lbl = batch["feat"], batch["label"]
                    out = user_model(feat)
                    loss = criterion_usr(out, lbl)
                    optimizer_usr.zero_grad()
                    loss.backward()
                    optimizer_usr.step()
                    t_loss += loss.item() * feat.size(0)
                    preds = out.argmax(1)
                    corr += (preds == lbl).sum().item()
                    tot += lbl.size(0)
                train_losses.append(t_loss / tot)
                train_accs.append(corr / tot)

                # Validate + alignment
                user_model.eval()
                v_loss = v_corr = v_tot = 0
                align_sum = align_tot = 0
                with torch.no_grad():
                    for batch in usr_val:
                        batch = {
                            k: v.to(device)
                            for k, v in batch.items()
                            if torch.is_tensor(v)
                        }
                        feat, lbl = batch["feat"], batch["label"]
                        out = user_model(feat)
                        loss = criterion_usr(out, lbl)
                        v_loss += loss.item() * feat.size(0)
                        preds = out.argmax(1)
                        v_corr += (preds == lbl).sum().item()
                        v_tot += lbl.size(0)
                        # alignment
                        P = feat[:, -2:].cpu().numpy()
                        Q = torch.softmax(out, 1).cpu().numpy()
                        M = 0.5 * (P + Q)
                        KL1 = np.sum(P * np.log((P + 1e-8) / (M + 1e-8)), axis=1)
                        KL2 = np.sum(Q * np.log((Q + 1e-8) / (M + 1e-8)), axis=1)
                        JS = 0.5 * (KL1 + KL2)
                        JSn = JS / np.log(2)
                        align = 1 - JSn
                        align_sum += align.sum()
                        align_tot += feat.size(0)
                val_losses.append(v_loss / v_tot)
                val_accs.append(v_corr / v_tot)
                alignment_scores.append(align_sum / align_tot)
                print(f"Epoch {epoch}: validation_loss = {val_losses[-1]:.4f}")

            # compute slope
            xs = np.arange(1, len(alignment_scores) + 1)
            slope = float(np.polyfit(xs, alignment_scores, 1)[0])

            # Test eval
            test_preds, test_gt = [], []
            user_model.eval()
            with torch.no_grad():
                for batch in usr_te:
                    batch = {
                        k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
                    }
                    feat, lbl = batch["feat"], batch["label"]
                    out = user_model(feat)
                    p = out.argmax(1).cpu().numpy()
                    test_preds.extend(p.tolist())
                    test_gt.extend(lbl.cpu().numpy().tolist())

            key = f"ai_bs_{ai_bs}_user_bs_{usr_bs}"
            experiment_data["teacher_prob_noise"][skey]["batch_size"][key] = {
                "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
                "losses": {
                    "train": np.array(train_losses),
                    "val": np.array(val_losses),
                },
                "alignment_scores": np.array(alignment_scores),
                "alignment_rate": slope,
                "predictions": np.array(test_preds),
                "ground_truth": np.array(test_gt),
            }

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
