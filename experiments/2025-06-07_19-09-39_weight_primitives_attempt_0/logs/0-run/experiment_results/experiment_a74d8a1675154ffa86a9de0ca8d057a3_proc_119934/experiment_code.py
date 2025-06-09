import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic data parameters
n_samples, n_test = 80, 20
n_components, dim = 30, 1024
lambda1, lr, epochs = 1e-2, 1e-2, 50

# generate ground truth and data
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.mm(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)
W_train, W_test = W_all[:n_samples], W_all[n_samples:]

# hyperparameter grid for Adam's beta1
beta1_list = [0.5, 0.7, 0.9, 0.99]

# define reconstruction losses
recon_fns = {
    "mse": lambda X, Y: ((X - Y) ** 2).mean(),
    "mae": lambda X, Y: (X - Y).abs().mean(),
    "huber": lambda X, Y: F.smooth_l1_loss(X, Y, reduction="mean"),
}

# prepare experiment data structure
experiment_data = {
    "reconstruction_loss": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for loss_name, recon_fn in recon_fns.items():
    ed = experiment_data["reconstruction_loss"]["synthetic"]
    for b1 in beta1_list:
        torch.manual_seed(0)
        D = nn.Parameter(torch.randn_like(D0))
        codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
        optimizer = torch.optim.Adam([D, codes_train], lr=lr, betas=(b1, 0.999))

        train_errs, val_errs = [], []
        train_losses, val_losses = [], []

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            W_hat = codes_train.mm(D)
            loss_recon = recon_fn(W_hat, W_train)
            loss_sparse = lambda1 * codes_train.abs().mean()
            (loss_recon + loss_sparse).backward()
            optimizer.step()

            with torch.no_grad():
                tr_err = (
                    ((W_hat - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
                )
                D_pinv = torch.pinverse(D)
                codes_test = W_test.mm(D_pinv)
                W_hat_test = codes_test.mm(D)
                vl_err = (
                    ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1))
                    .mean()
                    .item()
                )
                tr_loss = loss_recon.item()
                vl_loss = recon_fn(W_hat_test, W_test).item()

            train_errs.append(tr_err)
            val_errs.append(vl_err)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

        with torch.no_grad():
            D_pinv = torch.pinverse(D)
            W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()

        ed["metrics"]["train"].append(train_errs)
        ed["metrics"]["val"].append(val_errs)
        ed["losses"]["train"].append(train_losses)
        ed["losses"]["val"].append(val_losses)
        ed["predictions"].append(W_hat_test)
        ed["ground_truth"].append(W_test.cpu().numpy())

        print(f"Finished {loss_name} run for beta1={b1}")

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
