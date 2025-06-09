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

# sparsity regularization grid
lambda1_list = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

# prepare experiment data structure
experiment_data = {
    "sparsity_strength_ablation": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "sparsity": [],  # fraction of near-zero codes
            "dict_error": [],  # relative dictionary recovery error
            "predictions": [],
            "ground_truth": [],
        }
    }
}

for lam in lambda1_list:
    torch.manual_seed(0)
    # initialize parameters
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr)
    train_errs, val_errs = [], []
    train_losses, val_losses = [], []
    sparsities, dict_errors = [], []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        W_hat = codes_train.mm(D)
        loss_recon = ((W_hat - W_train) ** 2).mean()
        loss_sparse = lam * codes_train.abs().mean()
        loss = loss_recon + loss_sparse
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # reconstruction errors
            tr_err = ((W_hat - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
            D_pinv = torch.pinverse(D)
            codes_test = W_test.mm(D_pinv)
            W_hat_test = codes_test.mm(D)
            vl_err = (
                ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            )
            # losses (reconstruction only)
            tr_loss = loss_recon.item()
            vl_loss = ((W_hat_test - W_test) ** 2).mean().item()
            # code sparsity
            sparsity = (codes_train.detach().abs() < 1e-3).float().mean().item()
            # dictionary recovery error
            dict_err = (D.detach() - D0).norm() / D0.norm()
            dict_err = dict_err.item()

        train_errs.append(tr_err)
        val_errs.append(vl_err)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        sparsities.append(sparsity)
        dict_errors.append(dict_err)

    # final test predictions
    with torch.no_grad():
        D_pinv = torch.pinverse(D)
        W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()
    # store
    ed = experiment_data["sparsity_strength_ablation"]["synthetic"]
    ed["metrics"]["train"].append(train_errs)
    ed["metrics"]["val"].append(val_errs)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["val"].append(val_losses)
    ed["sparsity"].append(sparsities)
    ed["dict_error"].append(dict_errors)
    ed["predictions"].append(W_hat_test)
    ed["ground_truth"].append(W_test.cpu().numpy())
    print(f"Finished run for lambda1={lam}")

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
