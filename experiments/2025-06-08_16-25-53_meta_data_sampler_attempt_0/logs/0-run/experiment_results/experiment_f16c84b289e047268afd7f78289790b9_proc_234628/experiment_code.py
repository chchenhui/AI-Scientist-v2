import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic dataset
N = 1000
x = torch.rand(N, 1) * 6 - 3
y = torch.sin(x) + 0.1 * torch.randn_like(x)
x_train, y_train = x[:800], y[:800]
x_val, y_val = x[800:], y[800:]
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
x_val_tensor, y_val_tensor = x_val.to(device), y_val.to(device)


# Model definitions
class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


class DVN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


# Spearman correlation utility
def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


# Experiment data structure
experiment_data = {
    "meta_update_lr_tuning": {
        "synthetic": {
            "meta_lrs": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Hyperparameter grid
meta_lrs = np.logspace(-4, -1, num=6)
EPOCHS = 5
META_SAMPLE = 20

# Main tuning loop
for meta_lr in meta_lrs:
    print(f"\n=== Tuning META_UPDATE_LR = {meta_lr:.1e} ===")
    # Initialize models and optimizers
    main_model = PretrainModel().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-2)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-2)
    criterion_main = nn.MSELoss(reduction="none").to(device)
    criterion_dvn = nn.MSELoss(reduction="mean").to(device)

    # Containers for this meta_lr
    train_metrics, val_metrics = [], []
    train_losses, val_losses = [], []
    all_preds, all_truths = [], []

    for epoch in range(EPOCHS):
        # Train foundation model
        main_model.train()
        running_train = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = main_model(xb)
            loss_i = criterion_main(preds, yb)
            feats = loss_i.detach().unsqueeze(1)
            scores = dvn_model(feats).squeeze(1)
            weights = torch.softmax(scores, dim=0)
            loss = (weights * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()
            running_train += loss.item()
        tr_loss = running_train / len(train_loader)
        train_metrics.append(tr_loss)
        train_losses.append(tr_loss)

        # Validation
        main_model.eval()
        with torch.no_grad():
            val_preds = main_model(x_val_tensor)
            v_loss = criterion_main(val_preds, y_val_tensor).mean().item()
        print(f"MetaLR {meta_lr:.1e}, Epoch {epoch}: Val Loss = {v_loss:.4f}")
        val_metrics.append(v_loss)
        val_losses.append(v_loss)

        # Meta-update DVN
        features_list, contr_list = [], []
        base_state = main_model.state_dict()
        for idx in np.random.choice(len(x_train), META_SAMPLE, replace=False):
            xi = x_train[idx].unsqueeze(0).to(device)
            yi = y_train[idx].unsqueeze(0).to(device)
            with torch.no_grad():
                f0 = criterion_main(main_model(xi), yi).item()
            clone = PretrainModel().to(device)
            clone.load_state_dict(base_state)
            opt_clone = torch.optim.Adam(clone.parameters(), lr=float(meta_lr))
            clone.eval()
            with torch.no_grad():
                L0 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            clone.train()
            loss_ci = criterion_main(clone(xi), yi).mean()
            opt_clone.zero_grad()
            loss_ci.backward()
            opt_clone.step()
            clone.eval()
            with torch.no_grad():
                L1 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            features_list.append([f0])
            contr_list.append([L0 - L1])
        feats = torch.tensor(features_list, dtype=torch.float32).to(device)
        contrs = torch.tensor(contr_list, dtype=torch.float32).to(device)

        # Train DVN
        for _ in range(5):
            dvn_model.train()
            pred_c = dvn_model(feats)
            dvn_loss = criterion_dvn(pred_c, contrs)
            optimizer_dvn.zero_grad()
            dvn_loss.backward()
            optimizer_dvn.step()

        # Evaluate contribution correlation
        dvn_model.eval()
        with torch.no_grad():
            preds_np = dvn_model(feats).cpu().numpy().flatten()
        true_np = contrs.cpu().numpy().flatten()
        corr = spearman_corr(preds_np, true_np)
        print(f"MetaLR {meta_lr:.1e}, Epoch {epoch}: Spearman = {corr:.4f}")
        all_preds.append(preds_np)
        all_truths.append(true_np)

    # Store results
    d = experiment_data["meta_update_lr_tuning"]["synthetic"]
    d["meta_lrs"].append(float(meta_lr))
    d["metrics"]["train"].append(train_metrics)
    d["metrics"]["val"].append(val_metrics)
    d["losses"]["train"].append(train_losses)
    d["losses"]["val"].append(val_losses)
    d["predictions"].append(all_preds)
    d["ground_truth"].append(all_truths)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
