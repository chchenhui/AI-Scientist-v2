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
lambda1 = 1e-2  # sparsity weight
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

# orthogonality penalty grid
lambda2_list = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

# prepare experiment data structure
experiment_data = {
    "dict_orth": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# identity for orth penalty
I_comp = torch.eye(n_components, device=device)

for lam2 in lambda2_list:
    # reproducible init
    torch.manual_seed(0)
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr)

    train_errs, val_errs = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        W_hat = codes_train.mm(D)
        loss_recon = ((W_hat - W_train) ** 2).mean()
        loss_sparse = lambda1 * codes_train.abs().mean()
        orth_pen = ((D.mm(D.t()) - I_comp) ** 2).sum()
        loss = loss_recon + loss_sparse + lam2 * orth_pen
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            tr_err = ((W_hat - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
            D_pinv = torch.pinverse(D)
            codes_test = W_test.mm(D_pinv)
            W_hat_test = codes_test.mm(D)
            vl_err = (
                ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            )

            train_errs.append(tr_err)
            val_errs.append(vl_err)
            train_losses.append(loss_recon.item())
            val_losses.append(((W_hat_test - W_test) ** 2).mean().item())

    # final test predictions
    with torch.no_grad():
        D_pinv = torch.pinverse(D)
        W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()

    # store results
    ed = experiment_data["dict_orth"]["synthetic"]
    ed["metrics"]["train"].append(train_errs)
    ed["metrics"]["val"].append(val_errs)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["val"].append(val_losses)
    ed["predictions"].append(W_hat_test)
    ed["ground_truth"].append(W_test.cpu().numpy())

    print(f"Finished run for lambda2={lam2}")

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
