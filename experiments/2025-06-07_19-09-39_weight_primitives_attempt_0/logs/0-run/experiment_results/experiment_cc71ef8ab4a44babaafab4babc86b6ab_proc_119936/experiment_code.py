import os
import torch
import torch.nn as nn
import numpy as np
from itertools import product

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

# initialization methods
init_methods = {
    "normal": lambda x: x.data.normal_(),
    "xavier_uni": lambda x: torch.nn.init.xavier_uniform_(x),
    "orthogonal": lambda x: torch.nn.init.orthogonal_(x),
    "zeros": lambda x: x.data.zero_(),
}

# prepare experiment data structure
experiment_data = {
    "initialization": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "init_schemes": [],
        }
    }
}

# ablation over init combinations
for init_D, init_codes in product(init_methods.keys(), repeat=2):
    torch.manual_seed(0)
    # initialize parameters
    D = nn.Parameter(torch.empty(n_components, dim, device=device))
    init_methods[init_D](D)
    codes_train = nn.Parameter(torch.empty(n_samples, n_components, device=device))
    init_methods[init_codes](codes_train)
    optimizer = torch.optim.Adam([D, codes_train], lr=lr, betas=(0.9, 0.999))

    train_errs, val_errs = [], []
    train_losses, val_losses = [], []
    # training
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        W_hat = codes_train.mm(D)
        loss_recon = ((W_hat - W_train) ** 2).mean()
        loss_sparse = lambda1 * codes_train.abs().mean()
        (loss_recon + loss_sparse).backward()
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

    # final predictions
    with torch.no_grad():
        D_pinv = torch.pinverse(D)
        W_pred = (W_test.mm(D_pinv)).mm(D).cpu().numpy()
        W_gt = W_test.cpu().numpy()

    # store results
    ed = experiment_data["initialization"]["synthetic"]
    ed["metrics"]["train"].append(train_errs)
    ed["metrics"]["val"].append(val_errs)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["val"].append(val_losses)
    ed["predictions"].append(W_pred)
    ed["ground_truth"].append(W_gt)
    ed["init_schemes"].append({"D": init_D, "codes": init_codes})
    print(f"Finished init_D={init_D}, init_codes={init_codes}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
