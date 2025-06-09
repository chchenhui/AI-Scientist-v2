import os
import torch
import torch.nn as nn
import numpy as np

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic dataset params
n_samples, n_test = 80, 20
n_components, dim = 30, 1024
lambda1, lr, epochs = 1e-2, 1e-2, 50

# generate data
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.mm(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)
W_train, W_test = W_all[:n_samples], W_all[n_samples:]

# hyperparameter grid for L2 on D
weight_decays = [1e-5, 1e-4, 1e-3, 1e-2]

# initialize experiment storage
experiment_data = {
    "weight_decay": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": None,
        }
    }
}

for wd in weight_decays:
    # reinit parameters
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam(
        [
            {"params": [D], "lr": lr, "weight_decay": wd},
            {"params": [codes_train], "lr": lr, "weight_decay": 0},
        ]
    )
    metrics_train, metrics_val = [], []
    losses_train, losses_val = [], []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        W_hat_train = codes_train.mm(D)
        loss_recon = ((W_hat_train - W_train) ** 2).mean()
        loss_sparse = lambda1 * codes_train.abs().mean()
        (loss_recon + loss_sparse).backward()
        optimizer.step()

        with torch.no_grad():
            train_err = (
                ((W_hat_train - W_train).norm(dim=1) / W_train.norm(dim=1))
                .mean()
                .item()
            )
            D_pinv = torch.pinverse(D)
            codes_test = W_test.mm(D_pinv)
            W_hat_test = codes_test.mm(D)
            val_err = (
                ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            )
            val_loss = ((W_hat_test - W_test) ** 2).mean().item()

        metrics_train.append(train_err)
        metrics_val.append(val_err)
        losses_train.append(loss_recon.item())
        losses_val.append(val_loss)

    # store run results
    exp = experiment_data["weight_decay"]["synthetic"]
    exp["metrics"]["train"].append(metrics_train)
    exp["metrics"]["val"].append(metrics_val)
    exp["losses"]["train"].append(losses_train)
    exp["losses"]["val"].append(losses_val)
    exp["predictions"].append(W_hat_test.cpu().numpy())
    if exp["ground_truth"] is None:
        exp["ground_truth"] = W_test.cpu().numpy()

# convert lists to numpy arrays
exp = experiment_data["weight_decay"]["synthetic"]
exp["metrics"]["train"] = np.array(exp["metrics"]["train"])
exp["metrics"]["val"] = np.array(exp["metrics"]["val"])
exp["losses"]["train"] = np.array(exp["losses"]["train"])
exp["losses"]["val"] = np.array(exp["losses"]["val"])
exp["predictions"] = np.stack(exp["predictions"])
# ground_truth is already a NumPy array

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
