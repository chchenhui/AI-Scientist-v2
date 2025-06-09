import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Synthetic data
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


# Losses and utility
criterion_main = nn.MSELoss(reduction="none").to(device)
criterion_dvn = nn.MSELoss(reduction="mean").to(device)


def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


# Hyperparameter grid
wd_list = [0.0, 1e-5, 1e-4, 1e-3]
EPOCHS = 5
META_SAMPLE = 20

# Experiment storage
experiment_data = {
    "weight_decay": {
        "synthetic": {
            "weight_decay_main": [],
            "weight_decay_dvn": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Grid search
for wd_main in wd_list:
    for wd_dvn in wd_list:
        print(f"=== wd_main={wd_main}, wd_dvn={wd_dvn} ===")
        main_model = PretrainModel().to(device)
        dvn_model = DVN().to(device)
        optimizer_main = torch.optim.Adam(
            main_model.parameters(), lr=1e-2, weight_decay=wd_main
        )
        optimizer_dvn = torch.optim.Adam(
            dvn_model.parameters(), lr=1e-2, weight_decay=wd_dvn
        )

        run_train_metrics, run_val_metrics = [], []
        run_train_losses, run_val_losses = [], []
        run_preds, run_gts = [], []

        for epoch in range(EPOCHS):
            # --- Train main model ---
            main_model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = main_model(xb)
                loss_i = criterion_main(pred, yb)
                feats = loss_i.detach().unsqueeze(1)
                scores = dvn_model(feats).squeeze(1)
                w = torch.softmax(scores, dim=0)
                loss = (w * loss_i).sum()
                optimizer_main.zero_grad()
                loss.backward()
                optimizer_main.step()
                running_loss += loss.item()
            avg_train = running_loss / len(train_loader)
            run_train_metrics.append(avg_train)
            run_train_losses.append(avg_train)

            # --- Validation ---
            main_model.eval()
            with torch.no_grad():
                vp = main_model(x_val_tensor)
                avg_val = criterion_main(vp, y_val_tensor).mean().item()
            print(
                f"Run(wd_m={wd_main},wd_d={wd_dvn}) Epoch {epoch}: val_loss={avg_val:.4f}"
            )
            run_val_metrics.append(avg_val)
            run_val_losses.append(avg_val)

            # --- Meta-update DVN ---
            feats_list, contr_list = [], []
            base_state = main_model.state_dict()
            for idx in np.random.choice(len(x_train), META_SAMPLE, replace=False):
                xi = x_train[idx].unsqueeze(0).to(device)
                yi = y_train[idx].unsqueeze(0).to(device)
                with torch.no_grad():
                    fval = criterion_main(main_model(xi), yi).item()
                # one-step clone
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
                feats_list.append([fval])
                contr_list.append([L0 - L1])

            feats = torch.tensor(feats_list, dtype=torch.float32).to(device)
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
                pred_np = dvn_model(feats).cpu().numpy().flatten()
            true_np = contrs.cpu().numpy().flatten()
            corr = spearman_corr(pred_np, true_np)
            print(f"Epoch {epoch}: Spearman corr = {corr:.4f}")
            run_preds.append(pred_np)
            run_gts.append(true_np)

        # Save run results
        sd = experiment_data["weight_decay"]["synthetic"]
        sd["weight_decay_main"].append(wd_main)
        sd["weight_decay_dvn"].append(wd_dvn)
        sd["metrics"]["train"].append(run_train_metrics)
        sd["metrics"]["val"].append(run_val_metrics)
        sd["losses"]["train"].append(run_train_losses)
        sd["losses"]["val"].append(run_val_losses)
        sd["predictions"].append(run_preds)
        sd["ground_truth"].append(run_gts)

# Final save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
