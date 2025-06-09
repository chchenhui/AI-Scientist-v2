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

# fixed settings
n_samples = 80
n_test = 20
dim = 1024
lambda1 = 1e-2
lr = 1e-2
epochs = 50
n_components_list = [20, 30, 40, 50]

# initialize experiment data
experiment_data = {
    "n_components": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# generate one synthetic dataset using the largest dict size
max_components = max(n_components_list)
torch.manual_seed(0)
D0_full = torch.randn(max_components, dim, device=device)
codes0_full = (
    torch.rand(n_samples + n_test, max_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, max_components, device=device)
W_all = codes0_full.mm(D0_full) + 0.01 * torch.randn(
    n_samples + n_test, dim, device=device
)
W_train = W_all[:n_samples]
W_test = W_all[n_samples:]
W_test_np = W_test.cpu().numpy()

# hyperparameter sweep
for n_components in n_components_list:
    print(f"=== Tuning n_components = {n_components} ===")
    # initialize model params
    D = nn.Parameter(torch.randn(n_components, dim, device=device))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr)

    # logs for this setting
    train_errs, val_errs = [], []
    train_losses, val_losses = [], []

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
            train_errs.append(train_err)
            val_errs.append(val_err)
            train_losses.append(loss_recon.item())
            val_losses.append(((W_hat_test - W_test) ** 2).mean().item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"n_comp={n_components} Epoch {epoch}: val_err={val_err:.4f}")

    # store results
    exp = experiment_data["n_components"]["synthetic"]
    exp["metrics"]["train"].append(train_errs)
    exp["metrics"]["val"].append(val_errs)
    exp["losses"]["train"].append(train_losses)
    exp["losses"]["val"].append(val_losses)
    exp["predictions"].append(W_hat_test.cpu().numpy())
    exp["ground_truth"].append(W_test_np)

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
