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

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)


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
    def __init__(self, feat, label, p_ai):
        self.X = torch.from_numpy(feat).float()
        self.y = torch.from_numpy(label).long()
        self.p_ai = torch.from_numpy(p_ai).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i], "p_ai": self.p_ai[i]}


# Model definitions
class AIModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class UserModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# Hyperparameters
ai_batch_sizes = [16, 32, 64]
usr_batch_sizes = [16, 32, 64]
dataset_configs = [
    {"name": "D2_noise0.1_scale1.0", "D": 2, "noise": 0.1, "scale": 1.0},
    {"name": "D5_noise0.5_scale2.0", "D": 5, "noise": 0.5, "scale": 2.0},
    {"name": "D10_noise1.0_scale0.5", "D": 10, "noise": 1.0, "scale": 0.5},
]

# Container for experiment data
experiment_data = {"synthetic_diversity": {}}

for cfg in dataset_configs:
    name, D, noise, scale = cfg["name"], cfg["D"], cfg["noise"], cfg["scale"]
    # Generate synthetic data
    N = 2000
    X = np.random.randn(N, D)
    w_true = scale * np.random.randn(D)
    b_true = 0.5
    logits = X.dot(w_true) + b_true + noise * np.random.randn(N)
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(N) < probs).astype(int)
    # Split
    idx = np.random.permutation(N)
    tr_idx, vl_idx, ts_idx = idx[:1200], idx[1200:1500], idx[1500:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_vl, y_vl = X[vl_idx], y[vl_idx]
    X_ts, y_ts = X[ts_idx], y[ts_idx]
    # Normalize
    mean, std = X_tr.mean(0), X_tr.std(0) + 1e-6
    X_tr = (X_tr - mean) / std
    X_vl = (X_vl - mean) / std
    X_ts = (X_ts - mean) / std

    experiment_data["synthetic_diversity"][name] = {"batch_size": {}}

    # Train AI model for each batch size
    for ai_bs in ai_batch_sizes:
        # AI DataLoaders
        ai_tr_loader = DataLoader(SimpleDS(X_tr, y_tr), batch_size=ai_bs, shuffle=True)
        ai_vl_loader = DataLoader(SimpleDS(X_vl, y_vl), batch_size=ai_bs)
        # Build and train AI
        ai_model = AIModel(D, 16, 2).to(device)
        opt_ai = optim.Adam(ai_model.parameters(), lr=1e-2)
        crit_ai = nn.CrossEntropyLoss()
        for epoch in range(15):
            ai_model.train()
            for batch in ai_tr_loader:
                xb = batch["x"].to(device)
                yb = batch["y"].to(device)
                out = ai_model(xb)
                loss = crit_ai(out, yb)
                opt_ai.zero_grad()
                loss.backward()
                opt_ai.step()
        # Gather AI predictions
        ai_model.eval()
        with torch.no_grad():
            X_all = torch.from_numpy(np.vstack([X_tr, X_vl, X_ts])).float().to(device)
            logits_all = ai_model(X_all)
            probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
        p_tr = probs_all[: len(X_tr)]
        p_vl = probs_all[len(X_tr) : len(X_tr) + len(X_vl)]
        p_ts = probs_all[-len(X_ts) :]
        f_tr = p_tr.argmax(1)
        f_vl = p_vl.argmax(1)
        f_ts = p_ts.argmax(1)
        # Prepare user features
        X_utr = np.hstack([X_tr, p_tr])
        X_uvl = np.hstack([X_vl, p_vl])
        X_uts = np.hstack([X_ts, p_ts])

        for usr_bs in usr_batch_sizes:
            # DataLoaders for user model
            usr_tr_loader = DataLoader(
                UserDS(X_utr, f_tr, p_tr), batch_size=usr_bs, shuffle=True
            )
            usr_vl_loader = DataLoader(UserDS(X_uvl, f_vl, p_vl), batch_size=usr_bs)
            usr_ts_loader = DataLoader(UserDS(X_uts, f_ts, p_ts), batch_size=usr_bs)

            user_model = UserModel(D + 2, 8, 2).to(device)
            opt_usr = optim.Adam(user_model.parameters(), lr=1e-2)
            crit_usr = nn.CrossEntropyLoss()

            train_accs, val_accs = [], []
            train_losses, val_losses = [], []
            alignment_scores = []

            # Train user model with alignment metric
            epochs = 20
            for ep in range(epochs):
                # Training
                user_model.train()
                t_loss, t_corr, t_tot = 0.0, 0, 0
                for batch in usr_tr_loader:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    out = user_model(batch["feat"])
                    loss = crit_usr(out, batch["label"])
                    opt_usr.zero_grad()
                    loss.backward()
                    opt_usr.step()
                    t_loss += loss.item() * batch["feat"].size(0)
                    preds = out.argmax(1)
                    t_corr += (preds == batch["label"]).sum().item()
                    t_tot += batch["feat"].size(0)
                train_losses.append(t_loss / t_tot)
                train_accs.append(t_corr / t_tot)

                # Validation + Alignment
                user_model.eval()
                v_loss, v_corr, v_tot = 0.0, 0, 0
                p_list, q_list = [], []
                with torch.no_grad():
                    for batch in usr_vl_loader:
                        batch = {
                            k: v.to(device)
                            for k, v in batch.items()
                            if isinstance(v, torch.Tensor)
                        }
                        out = user_model(batch["feat"])
                        loss = crit_usr(out, batch["label"])
                        preds = out.argmax(1)
                        v_loss += loss.item() * batch["feat"].size(0)
                        v_corr += (preds == batch["label"]).sum().item()
                        v_tot += batch["feat"].size(0)
                        q_prob = torch.softmax(out, dim=1).cpu().numpy()
                        p_list.append(batch["p_ai"].cpu().numpy())
                        q_list.append(q_prob)
                val_losses.append(v_loss / v_tot)
                val_accs.append(v_corr / v_tot)

                # Compute Jensen-Shannon divergence and alignment score
                p_arr = np.vstack(p_list)
                q_arr = np.vstack(q_list)
                m = 0.5 * (p_arr + q_arr)
                kl1 = np.sum(p_arr * np.log2((p_arr + 1e-12) / (m + 1e-12)), axis=1)
                kl2 = np.sum(q_arr * np.log2((q_arr + 1e-12) / (m + 1e-12)), axis=1)
                jsd = 0.5 * (kl1 + kl2)
                align_score = 1.0 - jsd  # normalized in [0,1]
                epoch_align = np.mean(align_score)
                alignment_scores.append(epoch_align)

                print(
                    f"Epoch {ep+1}: validation_loss = {val_losses[-1]:.4f}, alignment_score = {epoch_align:.4f}"
                )

            # Compute alignment rate (slope)
            x = np.arange(1, epochs + 1)
            slope = np.polyfit(x, alignment_scores, 1)[0]

            # Test predictions
            test_preds, test_gt = [], []
            user_model.eval()
            with torch.no_grad():
                for batch in usr_ts_loader:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    out = user_model(batch["feat"])
                    test_preds.extend(out.argmax(1).cpu().numpy().tolist())
                    test_gt.extend(batch["label"].cpu().numpy().tolist())

            key = f"ai_bs_{ai_bs}_user_bs_{usr_bs}"
            experiment_data["synthetic_diversity"][name]["batch_size"][key] = {
                "metrics": {
                    "train_accs": np.array(train_accs),
                    "val_accs": np.array(val_accs),
                    "alignment_scores": np.array(alignment_scores),
                },
                "losses": {
                    "train": np.array(train_losses),
                    "val": np.array(val_losses),
                },
                "alignment_rate": slope,
                "predictions": np.array(test_preds),
                "ground_truth": np.array(test_gt),
            }

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
