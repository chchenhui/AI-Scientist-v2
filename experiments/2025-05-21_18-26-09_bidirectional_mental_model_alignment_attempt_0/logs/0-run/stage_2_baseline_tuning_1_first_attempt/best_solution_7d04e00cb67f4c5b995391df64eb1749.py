import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Synthetic data generation
N_train, N_val, D, C = 1000, 200, 10, 3
W_true = np.random.randn(D, C)
x_train = np.random.randn(N_train, D)
x_val = np.random.randn(N_val, D)
y_train = np.argmax(x_train @ W_true + 0.1 * np.random.randn(N_train, C), axis=1)
y_val = np.argmax(x_val @ W_true + 0.1 * np.random.randn(N_val, C), axis=1)
mean = x_train.mean(axis=0)
std = x_train.std(axis=0) + 1e-8
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std

# DataLoaders
train_dataset = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
)
val_dataset = TensorDataset(
    torch.tensor(x_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


# Simple MLP definition
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Hyperparameter sweep over learning rates
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
num_epochs = 10

# Initialize experiment_data dict
experiment_data = {
    "learning_rate": {
        "synthetic": {
            "lrs": learning_rates,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

loss_fn = nn.CrossEntropyLoss()

for lr in learning_rates:
    # Reset model weights for fair comparison
    torch.manual_seed(0)
    ai_model = MLP(D, 32, C).to(device)
    user_model = MLP(D, 32, C).to(device)
    optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=lr)
    optimizer_user = torch.optim.Adam(user_model.parameters(), lr=lr)

    # Containers for this learning rate
    train_losses, val_losses = [], []
    train_aligns, val_aligns = [], []

    for epoch in range(1, num_epochs + 1):
        # Training
        ai_model.train()
        user_model.train()
        total_loss, total_align, n_samples = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Forward passes
            logits_ai = ai_model(xb)
            logits_user = user_model(xb)
            # Cross-entropy losses
            loss_ai = loss_fn(logits_ai, yb)
            loss_user = loss_fn(logits_user, yb)
            # Backprop AI
            optimizer_ai.zero_grad()
            loss_ai.backward()
            optimizer_ai.step()
            # Backprop user
            optimizer_user.zero_grad()
            loss_user.backward()
            optimizer_user.step()
            # Accumulate loss
            bs = yb.size(0)
            total_loss += loss_ai.item() * bs
            # Alignment (1 - JSD)
            P = F.softmax(logits_ai, dim=1)
            Q = F.softmax(logits_user, dim=1)
            M = 0.5 * (P + Q)
            kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
            kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
            jsd = 0.5 * (kl1 + kl2)
            total_align += torch.sum(1 - jsd).item()
            n_samples += bs
        train_losses.append(total_loss / len(train_dataset))
        train_aligns.append(total_align / n_samples)

        # Validation
        ai_model.eval()
        user_model.eval()
        v_loss, v_align, v_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits_ai = ai_model(xb)
                # loss
                v_loss += loss_fn(logits_ai, yb).item() * yb.size(0)
                # alignment
                P = F.softmax(logits_ai, dim=1)
                Q = F.softmax(user_model(xb), dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                v_align += torch.sum(1 - jsd).item()
                v_samples += yb.size(0)
        val_losses.append(v_loss / len(val_dataset))
        val_aligns.append(v_align / v_samples)
        print(
            f"LR {lr:.1e} Epoch {epoch}: val_loss = {val_losses[-1]:.4f}, val_align = {val_aligns[-1]:.4f}"
        )

    # Store per‐epoch metrics
    sd = experiment_data["learning_rate"]["synthetic"]
    sd["metrics"]["train"].append(train_aligns)
    sd["metrics"]["val"].append(val_aligns)
    sd["losses"]["train"].append(train_losses)
    sd["losses"]["val"].append(val_losses)

    # Final validation predictions & ground truth
    all_preds, all_gts = [], []
    ai_model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = torch.argmax(ai_model(xb), dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(yb.numpy())
    preds_arr = np.concatenate(all_preds, axis=0)
    gts_arr = np.concatenate(all_gts, axis=0)
    sd["predictions"].append(preds_arr)
    sd["ground_truth"].append(gts_arr)

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
