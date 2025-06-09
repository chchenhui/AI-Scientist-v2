import os
import torch
import torch.nn as nn
import numpy as np

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyperparameters
torch.manual_seed(0)
n_samples, n_test = 80, 20
total = n_samples + n_test
n_components, dim = 30, 1024
lambda1, lr = 1e-2, 1e-2
epochs = 50
beta1_list = [0.5, 0.7, 0.9, 0.99]


# code sampling functions
def sample_bernoulli_gaussian(total, k, device, sparsity=0.1):
    mask = (torch.rand(total, k, device=device) < sparsity).float()
    return mask * torch.randn(total, k, device=device)


def sample_bernoulli_laplace(total, k, device, sparsity=0.1):
    mask = (torch.rand(total, k, device=device) < sparsity).float()
    dist = torch.distributions.Laplace(0.0, 1.0)
    return mask * dist.sample((total, k)).to(device)


def sample_bernoulli_uniform(total, k, device, sparsity=0.1):
    mask = (torch.rand(total, k, device=device) < sparsity).float()
    return mask * (2 * torch.rand(total, k, device=device) - 1.0)


def sample_block_sparse(total, k, device, block_size=5, blocks_per_sample=2):
    codes = torch.zeros(total, k, device=device)
    n_blocks = k // block_size
    for i in range(total):
        blocks = torch.randperm(n_blocks)[:blocks_per_sample]
        for b in blocks:
            s, e = b * block_size, b * block_size + block_size
            codes[i, s:e] = torch.randn(block_size, device=device)
    return codes


distributions = {
    "bernoulli_gaussian": sample_bernoulli_gaussian,
    "bernoulli_laplace": sample_bernoulli_laplace,
    "bernoulli_uniform": sample_bernoulli_uniform,
    "block_sparse": sample_block_sparse,
}

# ground-truth dictionary
D0 = torch.randn(n_components, dim, device=device)

# prepare experiment data structure
experiment_data = {"synthetic_code_distribution": {}}
for name in distributions:
    experiment_data["synthetic_code_distribution"][name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

# run ablation
for name, sampler in distributions.items():
    # synthesize data
    codes0 = sampler(total, n_components, device)
    W_all = codes0.mm(D0) + 0.01 * torch.randn(total, dim, device=device)
    W_train, W_test = W_all[:n_samples], W_all[n_samples:]
    print(f"Dataset: {name}")

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

        # final test predictions
        with torch.no_grad():
            D_pinv = torch.pinverse(D)
            W_hat_test = (W_test.mm(D_pinv)).mm(D).cpu().numpy()

        ed = experiment_data["synthetic_code_distribution"][name]
        ed["metrics"]["train"].append(train_errs)
        ed["metrics"]["val"].append(val_errs)
        ed["losses"]["train"].append(train_losses)
        ed["losses"]["val"].append(val_losses)
        ed["predictions"].append(W_hat_test)
        ed["ground_truth"].append(W_test.cpu().numpy())

        print(f"  Finished beta1={b1}")

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
