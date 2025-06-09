import os
import torch
import torch.nn as nn
import numpy as np

# working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# helper functions
def soft_threshold(x, thresh):
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)


def ista(D, W, lam, max_iter=50):
    n_comp = D.shape[0]
    A = D.matmul(D.t())
    v = torch.rand(n_comp, device=device)
    v /= v.norm()
    for _ in range(20):
        v = A.mv(v)
        v /= v.norm()
    L = v.dot(A.mv(v)).item()
    alpha = 1.0 / L
    X = torch.zeros((W.shape[0], n_comp), device=device)
    for _ in range(max_iter):
        grad = (X.matmul(D) - W).matmul(D.t())
        X = soft_threshold(X - alpha * grad, alpha * lam)
    return X


def fista(D, W, lam, max_iter=50):
    n_comp = D.shape[0]
    A = D.matmul(D.t())
    v = torch.rand(n_comp, device=device)
    v /= v.norm()
    for _ in range(20):
        v = A.mv(v)
        v /= v.norm()
    L = v.dot(A.mv(v)).item()
    alpha = 1.0 / L
    X = torch.zeros((W.shape[0], n_comp), device=device)
    Y = X.clone()
    t = 1.0
    for _ in range(max_iter):
        grad = (Y.matmul(D) - W).matmul(D.t())
        X_new = soft_threshold(Y - alpha * grad, alpha * lam)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        Y = X_new + ((t - 1) / t_new) * (X_new - X)
        X, t = X_new, t_new
    return X


def omp(D, W, k):
    # D: [n_comp, dim], W: [n_samples, dim]
    n_samples, n_comp = W.shape[0], D.shape[0]
    codes = torch.zeros((n_samples, n_comp), device=device)
    for i in range(n_samples):
        w = W[i]
        residual = w.clone()
        idxs = []
        x_S = None
        for _ in range(k):
            corr = torch.mv(D, residual)  # [n_comp]
            corr_abs = corr.abs()
            corr_abs[idxs] = 0
            j = int(torch.argmax(corr_abs))
            idxs.append(j)
            D_S = D[idxs]  # [|S|, dim]
            A = D_S.t()  # [dim, |S|]
            A_pinv = torch.pinverse(A)  # [|S|, dim]
            x_S = torch.mv(A_pinv, w)  # [|S|]
            residual = w - x_S.unsqueeze(0).mm(D_S).squeeze(0)
        if x_S is not None:
            codes[i, idxs] = x_S
    return codes


# synthetic data parameters
n_samples, n_test = 80, 20
n_components, dim = 30, 1024
lambda1, lr = 1e-2, 1e-2
epochs = 50
k_omp = max(1, int(0.1 * n_components))

# generate synthetic data
torch.manual_seed(0)
D0 = torch.randn(n_components, dim, device=device)
codes0 = (
    torch.rand(n_samples + n_test, n_components, device=device) < 0.1
).float() * torch.randn(n_samples + n_test, n_components, device=device)
W_all = codes0.matmul(D0) + 0.01 * torch.randn(n_samples + n_test, dim, device=device)
W_train, W_test = W_all[:n_samples], W_all[n_samples:]

# training setup
torch.manual_seed(0)
D = nn.Parameter(torch.randn_like(D0)).to(device)
codes_train = nn.Parameter(torch.randn(n_samples, n_components, device=device))
optimizer = torch.optim.Adam([D, codes_train], lr=lr, betas=(0.9, 0.999))

# define solvers
solver_funcs = {
    "pseudoinverse": lambda D_, W_: W_.matmul(torch.pinverse(D_)),
    "ista": lambda D_, W_: ista(D_, W_, lambda1, max_iter=50),
    "fista": lambda D_, W_: fista(D_, W_, lambda1, max_iter=50),
    "omp": lambda D_, W_: omp(D_, W_, k_omp),
}
solver_names = list(solver_funcs.keys())

# logs
train_errs, train_losses = [], []
val_errs = {name: [] for name in solver_names}
val_losses = {name: [] for name in solver_names}

# training loop
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    W_hat = codes_train.matmul(D)
    loss_recon = ((W_hat - W_train) ** 2).mean()
    loss_sparse = lambda1 * codes_train.abs().mean()
    (loss_recon + loss_sparse).backward()
    optimizer.step()

    with torch.no_grad():
        tr_err = ((W_hat - W_train).norm(dim=1) / W_train.norm(dim=1)).mean().item()
        train_errs.append(tr_err)
        train_losses.append(loss_recon.item())
        print(
            f"Epoch {epoch}: training_error = {tr_err:.4f}, training_loss = {loss_recon.item():.4f}"
        )
        for name, fn in solver_funcs.items():
            codes_test = fn(D, W_test)
            W_hat_test = codes_test.matmul(D)
            ve = ((W_hat_test - W_test).norm(dim=1) / W_test.norm(dim=1)).mean().item()
            vl = ((W_hat_test - W_test) ** 2).mean().item()
            val_errs[name].append(ve)
            val_losses[name].append(vl)
            print(f"  {name}: val_error = {ve:.4f}, val_loss = {vl:.6f}")

# finalize results
train_errs_np = np.array(train_errs)
train_losses_np = np.array(train_losses)
val_errs_np = {name: np.array(val_errs[name]) for name in solver_names}
val_losses_np = {name: np.array(val_losses[name]) for name in solver_names}

predictions_np = {}
for name, fn in solver_funcs.items():
    with torch.no_grad():
        codes_test = fn(D, W_test)
        predictions_np[name] = codes_test.matmul(D).cpu().numpy()
ground_truth_np = W_test.cpu().numpy()

experiment_data = {
    "test_time_solver_ablation": {
        "synthetic": {
            "solver_names": solver_names,
            "metrics": {"train_err": train_errs_np, "val_err": val_errs_np},
            "losses": {"train_loss": train_losses_np, "val_loss": val_losses_np},
            "predictions": predictions_np,
            "ground_truth": ground_truth_np,
        }
    }
}

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
