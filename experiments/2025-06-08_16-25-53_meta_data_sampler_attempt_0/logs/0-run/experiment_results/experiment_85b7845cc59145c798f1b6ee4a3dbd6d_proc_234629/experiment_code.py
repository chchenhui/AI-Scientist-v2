import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Set up working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic data
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


# Losses and experiment data scaffold
criterion_main = nn.MSELoss(reduction="none").to(device)
criterion_dvn = nn.MSELoss(reduction="mean").to(device)
T_list = [0.1, 0.5, 1.0, 2.0]
experiment_data = {
    "softmax_temperature": {
        "synthetic": {
            "T_values": T_list,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "correlation": [],
        }
    }
}


# Utility for Spearman correlation
def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


EPOCHS = 5
META_SAMPLE = 20

for T in T_list:
    print(f"\n=== Running with temperature T = {T} ===")
    # Re-init models & optimizers per T
    main_model = PretrainModel().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-2)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-2)

    # Containers for this T
    train_metrics, val_metrics = [], []
    train_losses, val_losses = [], []
    preds_list, gt_list, corr_list = [], [], []

    for epoch in range(EPOCHS):
        # 1) Train main model with weighted loss
        main_model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = main_model(xb)
            loss_i = criterion_main(preds, yb)  # per-sample losses
            feats = loss_i.detach().unsqueeze(1)
            scores = dvn_model(feats).squeeze(1)
            weights = torch.softmax(scores / T, dim=0)  # apply temperature
            loss = (weights * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)
        train_metrics.append(avg_train)
        train_losses.append(avg_train)

        # 2) Compute val loss
        main_model.eval()
        with torch.no_grad():
            val_pred = main_model(x_val_tensor)
            avg_val = criterion_main(val_pred, y_val_tensor).mean().item()
        val_metrics.append(avg_val)
        val_losses.append(avg_val)
        print(f" T={T}, Epoch {epoch} | Train {avg_train:.4f} | Val {avg_val:.4f}")

        # 3) Meta-update DVN
        features_list, contr_list = [], []
        base_state = main_model.state_dict()
        for idx in np.random.choice(len(x_train), META_SAMPLE, replace=False):
            xi = x_train[idx].unsqueeze(0).to(device)
            yi = y_train[idx].unsqueeze(0).to(device)
            # feature = current sample loss
            with torch.no_grad():
                fval = criterion_main(main_model(xi), yi).item()
            # one-step clone update to measure contribution
            clone = PretrainModel().to(device)
            clone.load_state_dict(base_state)
            opt_clone = torch.optim.Adam(clone.parameters(), lr=1e-2)
            with torch.no_grad():
                L0 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            clone.train()
            lc = criterion_main(clone(xi), yi).mean()
            opt_clone.zero_grad()
            lc.backward()
            opt_clone.step()
            clone.eval()
            with torch.no_grad():
                L1 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            contr = L0 - L1
            features_list.append([fval])
            contr_list.append([contr])

        feats = torch.tensor(features_list, dtype=torch.float32).to(device)
        contrs = torch.tensor(contr_list, dtype=torch.float32).to(device)
        # Train DVN
        for _ in range(5):
            dvn_model.train()
            pred_c = dvn_model(feats)
            loss_d = criterion_dvn(pred_c, contrs)
            optimizer_dvn.zero_grad()
            loss_d.backward()
            optimizer_dvn.step()
        # Eval DVN correlation
        dvn_model.eval()
        with torch.no_grad():
            p_np = dvn_model(feats).cpu().numpy().flatten()
        t_np = contrs.cpu().numpy().flatten()
        corr = spearman_corr(p_np, t_np)
        preds_list.append(p_np)
        gt_list.append(t_np)
        corr_list.append(corr)
        print(f" T={T}, Epoch {epoch} | DVN Spearman = {corr:.4f}")

    # Save results for this T
    sd = experiment_data["softmax_temperature"]["synthetic"]
    sd["metrics"]["train"].append(train_metrics)
    sd["metrics"]["val"].append(val_metrics)
    sd["losses"]["train"].append(train_losses)
    sd["losses"]["val"].append(val_losses)
    sd["predictions"].append(preds_list)
    sd["ground_truth"].append(gt_list)
    sd["correlation"].append(corr_list)

# Save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
