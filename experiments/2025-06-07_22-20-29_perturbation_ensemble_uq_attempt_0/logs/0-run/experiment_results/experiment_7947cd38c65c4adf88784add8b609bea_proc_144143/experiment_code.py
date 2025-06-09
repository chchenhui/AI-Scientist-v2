import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic data generation
def sample_data(N):
    N0 = N // 2
    N1 = N - N0
    d0 = np.clip(np.random.normal(0, 0.5, size=N0), 0, None)
    d1 = np.clip(np.random.normal(2, 1.0, size=N1), 0, None)
    xs = np.concatenate([d0, d1]).astype(np.float32).reshape(-1, 1)
    ys = np.concatenate([np.zeros(N0), np.ones(N1)]).astype(np.float32)
    idx = np.random.permutation(N)
    return xs[idx], ys[idx]


# Prepare datasets
x_train, y_train = sample_data(1000)
x_val, y_val = sample_data(200)
train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Hyperparameter tuning setup
gamma_list = [0.9, 0.95, 0.99]
epochs = 20
experiment_data = {"lr_scheduler_gamma": {"synthetic": {}}}

for gamma in gamma_list:
    key = str(gamma)
    # Initialize storage for this gamma
    experiment_data["lr_scheduler_gamma"]["synthetic"][key] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    # Model, optimizer, scheduler, loss
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses, all_preds, all_labels = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(yb.cpu().numpy())
        train_loss = np.mean(train_losses)
        train_preds = np.concatenate(all_preds)
        train_labels = np.concatenate(all_labels)
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_losses, v_preds, v_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(1)
                loss = loss_fn(logits, yb)
                val_losses.append(loss.item())
                v_preds.append(torch.sigmoid(logits).cpu().numpy())
                v_labels.append(yb.cpu().numpy())
        val_loss = np.mean(val_losses)
        val_preds = np.concatenate(v_preds)
        val_labels = np.concatenate(v_labels)
        val_auc = roc_auc_score(val_labels, val_preds)

        print(
            f"Gamma {gamma} Epoch {epoch}: val_loss = {val_loss:.4f}, val_auc = {val_auc:.4f}"
        )

        # Record metrics
        exp = experiment_data["lr_scheduler_gamma"]["synthetic"][key]
        exp["losses"]["train"].append(train_loss)
        exp["losses"]["val"].append(val_loss)
        exp["metrics"]["train"].append(train_auc)
        exp["metrics"]["val"].append(val_auc)
        exp["predictions"].append(val_preds)
        exp["ground_truth"].append(val_labels)

        # Step scheduler
        scheduler.step()

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
