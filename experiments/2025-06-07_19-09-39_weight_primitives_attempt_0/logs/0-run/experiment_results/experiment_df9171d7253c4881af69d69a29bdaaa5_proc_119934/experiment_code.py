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

# ablation: minibatch sizes (full batch + smaller)
batch_size_list = [n_samples, 40, 20, 10]

# prepare experiment data structure
experiment_data = {
    "mini_batch_size": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "batch_sizes": [],
        }
    }
}
synthetic = experiment_data["mini_batch_size"]["synthetic"]

for bs in batch_size_list:
    torch.manual_seed(0)
    # reinit params
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr, betas=(0.9, 0.999))
    train_errs, val_errs = [], []
    train_losses, val_losses = [], []
    synthetic["batch_sizes"].append(bs)
    # training loop
    for epoch in range(1, epochs + 1):
        if bs == n_samples:
            # full-batch update
            optimizer.zero_grad()
            W_hat = codes_train.mm(D)
            loss_recon = ((W_hat - W_train) ** 2).mean()
            loss_sparse = lambda1 * codes_train.abs().mean()
            (loss_recon + loss_sparse).backward()
            optimizer.step()
        else:
            # minibatch updates
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, bs):
                idx = perm[i : i + bs]
                optimizer.zero_grad()
                Wb = codes_train[idx].mm(D)
                lr_b = ((Wb - W_train[idx]) ** 2).mean()
                ls_b = lambda1 * codes_train[idx].abs().mean()
                (lr_b + ls_b).backward()
                optimizer.step()
        # compute metrics at epoch end
        with torch.no_grad():
            W_hat_full = codes_train.mm(D)
            tr_err = (
                ((W_hat_full - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
            )
            D_pinv = torch.pinverse(D)
            codes_test = W_test.mm(D_pinv)
            W_hat_test = codes_test.mm(D)
            vl_err = (
                ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            )
            tr_loss = ((W_hat_full - W_train) ** 2).mean().item()
            vl_loss = ((W_hat_test - W_test) ** 2).mean().item()
        train_errs.append(tr_err)
        val_errs.append(vl_err)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
    # final test predictions
    with torch.no_grad():
        D_pinv = torch.pinverse(D)
        W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()
    # store results
    synthetic["metrics"]["train"].append(train_errs)
    synthetic["metrics"]["val"].append(val_errs)
    synthetic["losses"]["train"].append(train_losses)
    synthetic["losses"]["val"].append(val_losses)
    synthetic["predictions"].append(W_hat_test)
    synthetic["ground_truth"].append(W_test.cpu().numpy())
    print(f"Finished run for batch_size={bs}")

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
