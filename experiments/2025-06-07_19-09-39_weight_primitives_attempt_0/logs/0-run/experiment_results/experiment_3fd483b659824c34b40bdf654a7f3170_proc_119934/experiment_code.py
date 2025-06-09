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
noise_scale = 0.01

# hyperparameter grid
beta1_list = [0.5, 0.7, 0.9, 0.99]

# initialize data
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)

# define noise generators
noise_types = {
    "gaussian": lambda: noise_scale
    * torch.randn((n_samples + n_test, dim), device=device),
    "laplace": lambda: torch.distributions.Laplace(0.0, noise_scale)
    .sample((n_samples + n_test, dim))
    .to(device),
    "cauchy": lambda: torch.distributions.Cauchy(0.0, noise_scale)
    .sample((n_samples + n_test, dim))
    .to(device),
}

# prepare experiment data structure
experiment_data = {"noise_distribution": {}}

for dist_name, noise_fn in noise_types.items():
    print(f"Starting ablation for noise: {dist_name}")
    # generate noisy data
    W_all = codes0.mm(D0) + noise_fn()
    W_train = W_all[:n_samples]
    W_test = W_all[n_samples:]
    # init storage
    experiment_data["noise_distribution"][dist_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    ed = experiment_data["noise_distribution"][dist_name]
    # run experiments for each beta1
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
            loss = loss_recon + loss_sparse
            loss.backward()
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
        with torch.no_grad():
            D_pinv = torch.pinverse(D)
            W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()
        # store results
        ed["metrics"]["train"].append(train_errs)
        ed["metrics"]["val"].append(val_errs)
        ed["losses"]["train"].append(train_losses)
        ed["losses"]["val"].append(val_losses)
        ed["predictions"].append(W_hat_test)
        ed["ground_truth"].append(W_test.cpu().numpy())
        print(f"  Finished beta1={b1}")
    print(f"Completed noise ablation for: {dist_name}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
