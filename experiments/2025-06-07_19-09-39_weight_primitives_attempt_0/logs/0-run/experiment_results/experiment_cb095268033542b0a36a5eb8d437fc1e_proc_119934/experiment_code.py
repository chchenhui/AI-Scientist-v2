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

# Synthetic and training parameters
n_samples, n_test = 80, 20
n_components, dim = 30, 1024
lambda1, lr, epochs = 1e-2, 1e-2, 50
beta1_list = [0.5, 0.7, 0.9, 0.99]

# Dataset configurations for ablation
datasets = {
    "ds1": {"seed": 0, "noise": 0.01, "sparsity": 0.1},
    "ds2": {"seed": 1, "noise": 0.05, "sparsity": 0.2},
    "ds3": {"seed": 2, "noise": 0.1, "sparsity": 0.3},
}

# Initialize experiment storage
experiment_data = {"multi_synthetic": {}}

for name, cfg in datasets.items():
    # prepare storage
    experiment_data["multi_synthetic"][name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "params": cfg,
    }
    # generate synthetic data
    torch.manual_seed(cfg["seed"])
    D0 = torch.randn(n_components, dim, device=device)
    codes0 = (
        torch.rand(n_samples + n_test, n_components, device=device) < cfg["sparsity"]
    ).float() * torch.randn(n_samples + n_test, n_components, device=device)
    W_all = codes0.mm(D0) + cfg["noise"] * torch.randn(
        n_samples + n_test, dim, device=device
    )
    W_train, W_test = W_all[:n_samples], W_all[n_samples:]
    # grid search over beta1 values
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
            loss_recon = ((W_hat - W_train) ** 2).mean()
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
                train_errs.append(tr_err)
                val_errs.append(vl_err)
                train_losses.append(loss_recon.item())
                val_losses.append(((W_hat_test - W_test) ** 2).mean().item())
        # compute final predictions
        with torch.no_grad():
            D_pinv = torch.pinverse(D)
            final_pred = (W_test.mm(D_pinv)).mm(D).cpu().numpy()
        # store results
        ed = experiment_data["multi_synthetic"][name]
        ed["metrics"]["train"].append(train_errs)
        ed["metrics"]["val"].append(val_errs)
        ed["losses"]["train"].append(train_losses)
        ed["losses"]["val"].append(val_losses)
        ed["predictions"].append(final_pred)
        ed["ground_truth"].append(W_test.cpu().numpy())
        print(f"Finished dataset={name}, beta1={b1}")

# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
