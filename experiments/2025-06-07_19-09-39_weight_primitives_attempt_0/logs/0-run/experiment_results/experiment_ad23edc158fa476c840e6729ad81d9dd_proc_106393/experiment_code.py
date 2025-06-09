import os
import torch
import torch.nn as nn
import numpy as np

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
# Synthetic weight dataset
n_samples, n_test, n_components, dim = 80, 20, 30, 1024
lambda1, lr, epochs = 1e-2, 1e-2, 50
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.mm(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)
W_train, W_test = W_all[:n_samples], W_all[n_samples:]

# Hyperparameter sweep
beta2_values = [0.9, 0.99, 0.999]
experiment_data = {
    "adam_beta2": {
        "synthetic": {
            "beta2": beta2_values,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for b2 in beta2_values:
    # initialize parameters & optimizer
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr, betas=(0.9, b2))
    mt, mv, lt, lv = [], [], [], []
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
        mt.append(train_err)
        mv.append(val_err)
        lt.append(loss_recon.item())
        lv.append(((W_hat_test - W_test) ** 2).mean().item())
        print(f"β₂={b2:.3f} Epoch {epoch}: val_err={val_err:.4f}")
    # collect
    sd = experiment_data["adam_beta2"]["synthetic"]
    sd["metrics"]["train"].append(mt)
    sd["metrics"]["val"].append(mv)
    sd["losses"]["train"].append(lt)
    sd["losses"]["val"].append(lv)
    sd["predictions"].append(W_hat_test.cpu().numpy())
    sd["ground_truth"].append(W_test.cpu().numpy())

# convert to numpy arrays and save
sd = experiment_data["adam_beta2"]["synthetic"]
for k in ["train", "val"]:
    sd["metrics"][k] = np.array(sd["metrics"][k])
    sd["losses"][k] = np.array(sd["losses"][k])
sd["predictions"] = np.array(sd["predictions"])
sd["ground_truth"] = np.array(sd["ground_truth"])
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
