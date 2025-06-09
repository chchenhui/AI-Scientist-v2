import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Prepare working dir and device
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


# Losses
criterion_main = nn.MSELoss(reduction="none").to(device)
criterion_dvn = nn.MSELoss(reduction="mean").to(device)


# Spearman correlation
def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


# Hyperparameter sweep
EPOCHS = 5
META_SAMPLES = [5, 10, 20, 50, 100]

# Data structure for results
experiment_data = {
    "meta_sample_tuning": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "corr": [],
            "meta_sample_values": [],
        }
    }
}

for meta_sample in META_SAMPLES:
    print(f"=== Running META_SAMPLE = {meta_sample} ===")
    # Initialize models & optimizers
    main_model = PretrainModel().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-2)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-2)
    # Lists to collect run-level data
    tr_metrics, val_metrics = [], []
    tr_losses, val_losses = [], []
    all_preds, all_truths, all_corr = [], [], []

    # Training loop
    for epoch in range(EPOCHS):
        main_model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = main_model(xb)
            loss_i = criterion_main(preds, yb)  # per-sample
            feats = loss_i.detach().unsqueeze(1)
            scores = dvn_model(feats).squeeze(1)
            weights = torch.softmax(scores, dim=0)
            loss = (weights * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        tr_metrics.append(train_loss)
        tr_losses.append(train_loss)

        main_model.eval()
        with torch.no_grad():
            val_pred = main_model(x_val_tensor)
            val_loss = criterion_main(val_pred, y_val_tensor).mean().item()
        val_metrics.append(val_loss)
        val_losses.append(val_loss)
        print(f"META_SAMPLE={meta_sample} Epoch={epoch} Val Loss={val_loss:.4f}")

        # Meta-update for DVN
        features_list, contr_list = [], []
        base_state = main_model.state_dict()
        for idx in np.random.choice(len(x_train), meta_sample, replace=False):
            xi = x_train[idx].unsqueeze(0).to(device)
            yi = y_train[idx].unsqueeze(0).to(device)
            with torch.no_grad():
                fval = criterion_main(main_model(xi), yi).item()
            clone = PretrainModel().to(device)
            clone.load_state_dict(base_state)
            opt_clone = torch.optim.Adam(clone.parameters(), lr=1e-2)
            with torch.no_grad():
                L0 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            clone.train()
            loss_ci = criterion_main(clone(xi), yi).mean()
            opt_clone.zero_grad()
            loss_ci.backward()
            opt_clone.step()
            with torch.no_grad():
                L1 = criterion_main(clone(x_val_tensor), y_val_tensor).mean().item()
            features_list.append([fval])
            contr_list.append([L0 - L1])

        feats = torch.tensor(features_list, dtype=torch.float32).to(device)
        contrs = torch.tensor(contr_list, dtype=torch.float32).to(device)
        for _ in range(5):
            dvn_model.train()
            pc = dvn_model(feats)
            dvn_loss = criterion_dvn(pc, contrs)
            optimizer_dvn.zero_grad()
            dvn_loss.backward()
            optimizer_dvn.step()

        dvn_model.eval()
        with torch.no_grad():
            preds_np = dvn_model(feats).cpu().numpy().flatten()
        truth_np = contrs.cpu().numpy().flatten()
        corr = spearman_corr(preds_np, truth_np)
        all_preds.append(preds_np)
        all_truths.append(truth_np)
        all_corr.append(corr)
        print(f"Epoch={epoch} Spearman Corr={corr:.4f}")

    # Store this run's data
    ds = experiment_data["meta_sample_tuning"]["synthetic"]
    ds["metrics"]["train"].append(tr_metrics)
    ds["metrics"]["val"].append(val_metrics)
    ds["losses"]["train"].append(tr_losses)
    ds["losses"]["val"].append(val_losses)
    ds["predictions"].append(all_preds)
    ds["ground_truth"].append(all_truths)
    ds["corr"].append(all_corr)
    ds["meta_sample_values"].append(meta_sample)

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
