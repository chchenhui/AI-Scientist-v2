import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)
np.random.seed(0)

# Ground-truth linear model
true_w = torch.tensor([1.0, -1.0], device=device)
true_b = torch.tensor([0.5], device=device)


def sample_data(n):
    X = torch.rand(n, 2, device=device) * 10 - 5
    logits = X.matmul(true_w) + true_b
    probs = torch.sigmoid(logits)
    y = (probs > 0.5).float()
    return X, y


# Generate and normalize data
X_train_raw, y_train = sample_data(1000)
X_test_raw, y_test = sample_data(200)
mean, std = X_train_raw.mean(0), X_train_raw.std(0)
X_train = (X_train_raw - mean) / std
X_test = (X_test_raw - mean) / std

# Pre-generate rejuvenation candidates and select high-uncertainty points
X_cand_raw = torch.rand(5000, 2, device=device) * 10 - 5
X_cand = (X_cand_raw - mean) / std


class LogisticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.lin(x).squeeze(1)


# Initialize K models
K = 3
models, optimizers = [], []
for _ in range(K):
    m = LogisticModel().to(device)
    models.append(m)
    optimizers.append(torch.optim.SGD(m.parameters(), lr=0.1))

# Use first model's random init to pick entropic samples
with torch.no_grad():
    logits = models[0](X_cand)
    p = torch.sigmoid(logits)
    H = -p * torch.log(p + 1e-12) - (1 - p) * torch.log(1 - p + 1e-12)
    topk = torch.topk(H, 100).indices
X_rej_raw = X_cand_raw[topk]
X_rej = (X_rej_raw - mean) / std
with torch.no_grad():
    logits_true = X_rej_raw.matmul(true_w) + true_b
    y_rej = (torch.sigmoid(logits_true) > 0.5).float()

# DataLoaders
train_ds = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_ds = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=200)

# Prepare experiment data storage
experiment_data = {
    "synthetic_linear": {
        "metrics": {
            "train_loss": [],
            "val_loss": [],
            "std_original": [],
            "std_rejuvenated": [],
            "CGR": [],
        },
        "predictions": [],
        "ground_truth": y_test.cpu().numpy().tolist(),
    }
}

# Training and evaluation loop
epochs = 20
for epoch in range(epochs):
    # Train each model
    for m, opt in zip(models, optimizers):
        m.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = m(Xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * Xb.size(0)
    avg_train_loss = total_loss / len(train_ds)

    # Evaluate on original test
    accs, val_loss = [], 0.0
    with torch.no_grad():
        Xb, yb = next(iter(test_loader))
        Xb, yb = Xb.to(device), yb.to(device)
        for m in models:
            m.eval()
            logits = m(Xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            val_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            accs.append((preds == yb).float().mean().item())
    val_loss /= K
    std_orig = np.std(accs)

    # Evaluate on augmented test (with rejuvenation)
    accs_rej = []
    X_aug = torch.cat([X_test, X_rej], dim=0)
    y_aug = torch.cat([y_test, y_rej], dim=0)
    with torch.no_grad():
        for m in models:
            logits = m(X_aug)
            preds = (torch.sigmoid(logits) > 0.5).float()
            accs_rej.append((preds == y_aug).float().mean().item())
    std_rej = np.std(accs_rej)
    CGR = (std_rej - std_orig) / std_orig if std_orig != 0 else 0.0

    # Print and record
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, CGR = {CGR:.4f}")
    experiment_data["synthetic_linear"]["metrics"]["train_loss"].append(avg_train_loss)
    experiment_data["synthetic_linear"]["metrics"]["val_loss"].append(val_loss)
    experiment_data["synthetic_linear"]["metrics"]["std_original"].append(std_orig)
    experiment_data["synthetic_linear"]["metrics"]["std_rejuvenated"].append(std_rej)
    experiment_data["synthetic_linear"]["metrics"]["CGR"].append(CGR)

    # Record predictions of first model
    with torch.no_grad():
        logits = models[0](X_test)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy().tolist()
    experiment_data["synthetic_linear"]["predictions"].append(preds)

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
