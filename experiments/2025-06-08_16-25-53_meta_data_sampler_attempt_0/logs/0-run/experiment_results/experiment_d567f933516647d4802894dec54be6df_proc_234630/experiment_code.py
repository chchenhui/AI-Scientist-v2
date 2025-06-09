import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# prepare working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# reproducibility
torch.manual_seed(0)
np.random.seed(0)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic data
N = 1000
x = torch.rand(N, 1) * 6 - 3
y = torch.sin(x) + 0.1 * torch.randn_like(x)
x_train, y_train = x[:800], y[:800]
x_val, y_val = x[800:], y[800:]
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
x_val_tensor, y_val_tensor = x_val.to(device), y_val.to(device)


# models
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


# losses
criterion_main = nn.MSELoss(reduction="none").to(device)
criterion_dvn = nn.MSELoss(reduction="mean").to(device)


# spearman
def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


# hyperparameter tuning structure
experiment_data = {"ADAM_BETA2": {}}
beta2_list = [0.9, 0.98, 0.999]
EPOCHS = 5
META_SAMPLE = 20

for b2 in beta2_list:
    key = f"beta2_{b2}"
    # init storage
    data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # init models + optimizers
    main_model = PretrainModel().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-2, betas=(0.9, b2))
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-2, betas=(0.9, b2))

    for epoch in range(EPOCHS):
        # train main
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
        data["metrics"]["train"].append(tr_loss)
        data["losses"]["train"].append(tr_loss)
        # val
        main_model.eval()
        with torch.no_grad():
            val_preds = main_model(x_val_tensor)
            vl = criterion_main(val_preds, y_val_tensor).mean().item()
        data["metrics"]["val"].append(vl)
        data["losses"]["val"].append(vl)
        print(f"[{key}] Epoch {epoch} Train {tr_loss:.4f} Val {vl:.4f}")

        # meta-update DVN
        features_list, contr_list = [], []
        base_state = main_model.state_dict()
        for idx in np.random.choice(len(x_train), META_SAMPLE, replace=False):
            xi = x_train[idx].unsqueeze(0).to(device)
            yi = y_train[idx].unsqueeze(0).to(device)
            with torch.no_grad():
                f0 = criterion_main(main_model(xi), yi).item()
            clone = PretrainModel().to(device)
            clone.load_state_dict(base_state)
            opt_clone = torch.optim.Adam(clone.parameters(), lr=1e-2, betas=(0.9, b2))
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
        for _ in range(5):
            dvn_model.train()
            pred_c = dvn_model(feats)
            l_dvn = criterion_dvn(pred_c, contrs)
            optimizer_dvn.zero_grad()
            l_dvn.backward()
            optimizer_dvn.step()

        dvn_model.eval()
        with torch.no_grad():
            p_np = dvn_model(feats).cpu().numpy().flatten()
        t_np = contrs.cpu().numpy().flatten()
        corr = spearman_corr(p_np, t_np)
        print(f"[{key}] Epoch {epoch} Spearman Corr: {corr:.4f}")
        data["predictions"].append(p_np)
        data["ground_truth"].append(t_np)

    # store
    experiment_data["ADAM_BETA2"][key] = {"synthetic": data}

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
