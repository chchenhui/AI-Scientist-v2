# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic weight dataset
n_samples = 80
n_test = 20
n_components = 30
dim = 1024
lambda1 = 1e-2
lr = 1e-2
epochs = 50

# ground truth primitives and codes
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.mm(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)

# train/test split
W_train = W_all[:n_samples]
W_test = W_all[n_samples:]

# learnable params
D = nn.Parameter(torch.randn_like(D0))
codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))

optimizer = torch.optim.Adam([D, codes_train], lr=lr)

experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    W_hat_train = codes_train.mm(D)
    loss_recon = ((W_hat_train - W_train) ** 2).mean()
    loss_sparse = lambda1 * codes_train.abs().mean()
    loss = loss_recon + loss_sparse
    loss.backward()
    optimizer.step()

    # compute train error
    with torch.no_grad():
        train_err = (
            ((W_hat_train - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
        )
        # test codes via pinv
        D_pinv = torch.pinverse(D)
        codes_test = W_test.mm(D_pinv)
        W_hat_test = codes_test.mm(D)
        val_err = ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()

    experiment_data["synthetic"]["metrics"]["train"].append(train_err)
    experiment_data["synthetic"]["metrics"]["val"].append(val_err)
    experiment_data["synthetic"]["losses"]["train"].append(loss_recon.item())
    experiment_data["synthetic"]["losses"]["val"].append(
        ((W_hat_test - W_test) ** 2).mean().item()
    )

    print(f"Epoch {epoch}: validation_loss = {val_err:.4f}")

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
