import os
import torch
import torch.nn as nn
import numpy as np

# single‐file script
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

# generate ground-truth dictionary and codes
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)

# ablation: noise levels
noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05]

# storage lists
metrics_train_list = []
metrics_val_list = []
losses_train_list = []
losses_val_list = []
predictions_list = []
ground_truth_list = []
noise_list = []

for sigma in noise_levels:
    # regenerate noisy observations
    torch.manual_seed(0)
    noise = torch.randn(n_samples + n_test, dim, device=device)
    W_all = codes0.mm(D0) + sigma * noise
    W_train = W_all[:n_samples]
    W_test = W_all[n_samples:]

    # reinit model parameters
    torch.manual_seed(0)
    D = nn.Parameter(torch.randn_like(D0))
    codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
    optimizer = torch.optim.Adam([D, codes_train], lr=lr)  # default betas=(0.9,0.999)

    train_errs, val_errs = [], []
    train_losses, val_losses = [], []

    # training loop
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        W_hat = codes_train.mm(D)
        loss_recon = ((W_hat - W_train) ** 2).mean()
        loss_sparse = lambda1 * codes_train.abs().mean()
        loss = loss_recon + loss_sparse
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # train error
            tr_err = ((W_hat - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
            # validation error
            D_pinv = torch.pinverse(D)
            codes_test = W_test.mm(D_pinv)
            W_hat_test = codes_test.mm(D)
            vl_err = (
                ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            )
            # losses
            tr_loss = loss_recon.item()
            vl_loss = ((W_hat_test - W_test) ** 2).mean().item()

        train_errs.append(tr_err)
        val_errs.append(vl_err)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

    # final test reconstruction
    with torch.no_grad():
        D_pinv = torch.pinverse(D)
        W_hat_final = W_test.mm(D_pinv).mm(D)

    # store results
    metrics_train_list.append(train_errs)
    metrics_val_list.append(val_errs)
    losses_train_list.append(train_losses)
    losses_val_list.append(val_losses)
    predictions_list.append(W_hat_final.cpu().numpy())
    ground_truth_list.append(W_test.cpu().numpy())
    noise_list.append(sigma)

    print(f"Finished run for noise σ={sigma}")

# convert to numpy arrays
metrics_train_arr = np.array(metrics_train_list)  # shape (len(sigmas), epochs)
metrics_val_arr = np.array(metrics_val_list)
losses_train_arr = np.array(losses_train_list)
losses_val_arr = np.array(losses_val_list)
predictions_arr = np.stack(predictions_list)  # shape (len(sigmas), n_test, dim)
ground_truth_arr = np.stack(ground_truth_list)
noise_arr = np.array(noise_list)

# compile experiment data
experiment_data = {
    "synthetic_noise": {
        "synthetic": {
            "metrics": {"train": metrics_train_arr, "val": metrics_val_arr},
            "losses": {"train": losses_train_arr, "val": losses_val_arr},
            "predictions": predictions_arr,
            "ground_truth": ground_truth_arr,
            "noise_levels": noise_arr,
        }
    }
}

# save to file
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
