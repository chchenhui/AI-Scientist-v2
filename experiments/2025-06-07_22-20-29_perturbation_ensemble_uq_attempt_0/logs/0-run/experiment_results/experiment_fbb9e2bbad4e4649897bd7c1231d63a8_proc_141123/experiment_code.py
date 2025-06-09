import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic classification data
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=15, n_redundant=5, random_state=42
)
split = int(0.8 * X.shape[0])
X_train_np, X_val_np = X[:split], X[split:]
y_train_np, y_val_np = y[:split], y[split:]
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.long)
X_val = torch.tensor(X_val_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)


# Define a simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = MLP(input_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Prepare experiment data storage
experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

num_epochs = 10
K = 5  # number of perturbations


def compute_auc(X_all, y_all):
    X_all = X_all.to(device)
    with torch.no_grad():
        orig_logits = model(X_all)
        orig_preds = torch.argmax(orig_logits, dim=1).cpu().numpy()
    # generate perturbations
    X_rep = X_all.unsqueeze(1).repeat(1, K, 1)
    noise = torch.randn_like(X_rep) * 0.1
    X_pert = (X_rep + noise).view(-1, X_all.size(1))
    with torch.no_grad():
        logits_k = model(X_pert)
        preds_k = torch.argmax(logits_k, dim=1).view(-1, K).cpu().numpy()
    errors = (orig_preds != y_all.numpy()).astype(int)
    divergence = []
    for pk in preds_k:
        _, counts = np.unique(pk, return_counts=True)
        f_max = counts.max()
        divergence.append(1 - f_max / K)
    auc = roc_auc_score(errors, divergence)
    return auc, errors, divergence


# Training loop
for epoch in range(1, num_epochs + 1):
    # Train
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
    train_loss /= len(train_loader.dataset)
    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = criterion(model(Xb), yb)
            val_loss += loss.item() * Xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    # Compute detection AUCs
    train_auc, _, _ = compute_auc(X_train, y_train)
    val_auc, val_errors, val_div = compute_auc(X_val, y_val)
    experiment_data["synthetic"]["losses"]["train"].append(train_loss)
    experiment_data["synthetic"]["losses"]["val"].append(val_loss)
    experiment_data["synthetic"]["metrics"]["train"].append(train_auc)
    experiment_data["synthetic"]["metrics"]["val"].append(val_auc)

# Record final predictions and ground truth
experiment_data["synthetic"]["predictions"] = val_div
experiment_data["synthetic"]["ground_truth"] = val_errors

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)

# Visualization: histogram of divergence scores
plt.figure()
val_div = np.array(val_div)
val_errors = np.array(val_errors)
plt.hist(val_div[val_errors == 0], bins=20, alpha=0.5, label="Correct")
plt.hist(val_div[val_errors == 1], bins=20, alpha=0.5, label="Incorrect")
plt.legend()
plt.title("Divergence Distribution on Val Set")
plt.xlabel("Uncertainty Score (Divergence)")
plt.ylabel("Count")
plt.savefig(os.path.join(working_dir, "divergence_histogram.png"))

print(f"Final validation Hallucination Detection AUC-ROC: {val_auc:.4f}")
