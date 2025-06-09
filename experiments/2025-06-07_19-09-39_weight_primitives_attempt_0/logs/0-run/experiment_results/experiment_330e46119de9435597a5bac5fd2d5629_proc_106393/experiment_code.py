import os
import torch
import torch.nn as nn
import numpy as np

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# problem setup
n_samples, n_test = 80, 20
n_components, dim = 30, 1024
lambda1 = 1e-2
epochs = 50

# synthetic data (fixed)
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.mm(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)
W_train, W_test = W_all[:n_samples], W_all[n_samples:]

# hyperparameter sweep
lrs = [1e-3, 1e-2, 1e-1]
experiment_data = {
    "learning_rate_sweep": {
        "synthetic": {
            "lrs": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": None,
        }
    }
}

for lr in lrs:
    print(f"\n--- Training with learning rate = {lr} ---")
    # re-init model params
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr)

    train_errs, val_errs = [], []
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        W_hat_train = codes_train.mm(D)
        loss_recon = ((W_hat_train - W_train) ** 2).mean()
        loss_sparse = lambda1 * codes_train.abs().mean()
        loss = loss_recon + loss_sparse
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            te = (
                ((W_hat_train - W_train).norm(dim=1) / W_train.norm(dim=1))
                .mean()
                .item()
            )
            D_pinv = torch.pinverse(D)
            codes_test = W_test.mm(D_pinv)
            W_hat_test = codes_test.mm(D)
            ve = ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            vl = ((W_hat_test - W_test) ** 2).mean().item()

        train_errs.append(te)
        val_errs.append(ve)
        train_losses.append(loss_recon.item())
        val_losses.append(vl)

        print(f"LR {lr} Epoch {epoch}: val_err = {ve:.4f}")

    # record results for this lr
    experiment_data["learning_rate_sweep"]["synthetic"]["lrs"].append(lr)
    experiment_data["learning_rate_sweep"]["synthetic"]["metrics"]["train"].append(
        train_errs
    )
    experiment_data["learning_rate_sweep"]["synthetic"]["metrics"]["val"].append(
        val_errs
    )
    experiment_data["learning_rate_sweep"]["synthetic"]["losses"]["train"].append(
        train_losses
    )
    experiment_data["learning_rate_sweep"]["synthetic"]["losses"]["val"].append(
        val_losses
    )
    # final test predictions
    experiment_data["learning_rate_sweep"]["synthetic"]["predictions"].append(
        W_hat_test.cpu().numpy()
    )

# ground truth once
experiment_data["learning_rate_sweep"]["synthetic"][
    "ground_truth"
] = W_test.cpu().numpy()

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
