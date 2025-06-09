import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

# setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic data parameters
n_samples, n_test, n_components, dim = 80, 20, 30, 1024
lambda1, lr, epochs = 1e-2, 1e-2, 50

# generate ground truth and data
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.mm(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)
W_train, W_test = W_all[:n_samples], W_all[n_samples:]

# define learning‐rate schedules
schedule_configs = {
    "fixed": None,
    "step_decay": lambda opt: StepLR(opt, step_size=15, gamma=0.1),
    "exp_decay": lambda opt: ExponentialLR(opt, gamma=0.95),
    "cosine": lambda opt: CosineAnnealingLR(opt, T_max=epochs),
}

# prepare experiment data structure
experiment_data = {
    "lr_schedules": {
        "synthetic": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "schedules": [],
        }
    }
}

# run ablation
for schedule_name, sched_fn in schedule_configs.items():
    torch.manual_seed(0)
    # reinit model parameters
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr, betas=(0.9, 0.999))
    scheduler = sched_fn(optimizer) if sched_fn is not None else None

    train_errs, val_errs = [], []
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        W_hat = codes_train.mm(D)
        loss_recon = ((W_hat - W_train) ** 2).mean()
        loss_sparse = lambda1 * codes_train.abs().mean()
        (loss_recon + loss_sparse).backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
        W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()

    ed = experiment_data["lr_schedules"]["synthetic"]
    ed["metrics"]["train"].append(train_errs)
    ed["metrics"]["val"].append(val_errs)
    ed["losses"]["train"].append(train_losses)
    ed["losses"]["val"].append(val_losses)
    ed["predictions"].append(W_hat_test)
    ed["ground_truth"].append(W_test.cpu().numpy())
    ed["schedules"].append(schedule_name)
    print(f"Finished run for schedule={schedule_name}")

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
