# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic dataset
N = 1000
x = torch.rand(N, 1) * 6 - 3
y = torch.sin(x) + 0.1 * torch.randn_like(x)
x_train, y_train = x[:800], y[:800]
x_val, y_val = x[800:], y[800:]
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
x_val_tensor, y_val_tensor = x_val.to(device), y_val.to(device)


# Models
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


main_model = PretrainModel().to(device)
dvn_model = DVN().to(device)

optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-2)
optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-2)
criterion_main = nn.MSELoss(reduction="none").to(device)
criterion_dvn = nn.MSELoss(reduction="mean").to(device)


# Utility: spearman correlation via numpy
def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


# Experiment data storage
experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

EPOCHS = 5
META_SAMPLE = 20

for epoch in range(EPOCHS):
    # Train foundation model with DVN weights
    main_model.train()
    running_train_loss = 0.0
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
        running_train_loss += loss.item()
    train_loss = running_train_loss / len(train_loader)
    experiment_data["synthetic"]["metrics"]["train"].append(train_loss)
    experiment_data["synthetic"]["losses"]["train"].append(train_loss)

    # Validation loss
    main_model.eval()
    with torch.no_grad():
        val_preds = main_model(x_val_tensor)
        val_loss = criterion_main(val_preds, y_val_tensor).mean().item()
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    experiment_data["synthetic"]["metrics"]["val"].append(val_loss)
    experiment_data["synthetic"]["losses"]["val"].append(val_loss)

    # Meta-update DVN: measure actual contributions
    features_list, contr_list = [], []
    base_state = main_model.state_dict()
    for idx in np.random.choice(len(x_train), META_SAMPLE, replace=False):
        xi = x_train[idx].unsqueeze(0).to(device)
        yi = y_train[idx].unsqueeze(0).to(device)
        # feature = current loss
        with torch.no_grad():
            feature_val = criterion_main(main_model(xi), yi).item()
        # clone model for one-step update
        clone = PretrainModel().to(device)
        clone.load_state_dict(base_state)
        opt_clone = torch.optim.Adam(clone.parameters(), lr=1e-2)
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
        contr = L0 - L1
        features_list.append([feature_val])
        contr_list.append([contr])

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
    # Evaluate correlation
    dvn_model.eval()
    with torch.no_grad():
        preds_np = dvn_model(feats).cpu().numpy().flatten()
    true_np = contrs.cpu().numpy().flatten()
    corr = spearman_corr(preds_np, true_np)
    print(f"Epoch {epoch}: Contribution Prediction Correlation = {corr:.4f}")
    experiment_data["synthetic"]["predictions"].append(preds_np)
    experiment_data["synthetic"]["ground_truth"].append(true_np)

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
