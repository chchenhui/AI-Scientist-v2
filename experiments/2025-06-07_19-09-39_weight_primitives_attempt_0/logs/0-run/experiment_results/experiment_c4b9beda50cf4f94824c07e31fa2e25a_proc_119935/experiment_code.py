import os
import torch
import torch.nn as nn
import numpy as np

# setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic data parameters
n_samples = 80
n_test = 20
n_components = 30
dim = 1024
lambda1 = 1e-2
lr = 1e-2
epochs = 50

# generate ground truth and data
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.mm(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)
W_train = W_all[:n_samples]
W_test = W_all[n_samples:]

# ablation schedules: (code_updates, dict_updates)
ratio_pairs = [(1, 1), (5, 1), (10, 1), (1, 5), (1, 10)]

# prepare experiment data structure
experiment_data = {
    "alt_min_freq": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "ratios": [],
        }
    }
}

for c_steps, d_steps in ratio_pairs:
    torch.manual_seed(0)
    # initialize parameters
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    # separate optimizers
    opt_codes = torch.optim.Adam([codes_train], lr=lr, betas=(0.9, 0.999))
    opt_D = torch.optim.Adam([D], lr=lr, betas=(0.9, 0.999))
    # recorders
    train_errs, val_errs = [], []
    train_losses, val_losses = [], []
    # training loop
    for epoch in range(1, epochs + 1):
        # code updates
        for _ in range(c_steps):
            opt_codes.zero_grad()
            W_hat = codes_train.mm(D.detach())
            loss_recon = ((W_hat - W_train) ** 2).mean()
            loss_sparse = lambda1 * codes_train.abs().mean()
            (loss_recon + loss_sparse).backward()
            opt_codes.step()
        # dictionary updates
        for _ in range(d_steps):
            opt_D.zero_grad()
            W_hat = codes_train.detach().mm(D)
            loss_recon = ((W_hat - W_train) ** 2).mean()
            loss_sparse = lambda1 * codes_train.abs().mean()
            (loss_recon + loss_sparse).backward()
            opt_D.step()
        # compute metrics
        with torch.no_grad():
            W_hat_tr = codes_train.mm(D)
            tr_err = (
                ((W_hat_tr - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
            )
            loss_tr = ((W_hat_tr - W_train) ** 2).mean().item()
            D_pinv = torch.pinverse(D)
            codes_test = W_test.mm(D_pinv)
            W_hat_test = codes_test.mm(D)
            vl_err = (
                ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            )
            loss_val = ((W_hat_test - W_test) ** 2).mean().item()
        train_errs.append(tr_err)
        val_errs.append(vl_err)
        train_losses.append(loss_tr)
        val_losses.append(loss_val)
    # final predictions
    with torch.no_grad():
        D_pinv = torch.pinverse(D)
        W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()
    # store results
    ed = experiment_data["alt_min_freq"]["synthetic"]
    ed["metrics"]["train"].append(train_errs)
    ed["metrics"]["val"].append(val_errs)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["val"].append(val_losses)
    ed["predictions"].append(W_hat_test)
    ed["ground_truth"].append(W_test.cpu().numpy())
    ed["ratios"].append((c_steps, d_steps))
    print(f"Finished run for code:dict updates = {c_steps}:{d_steps}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
