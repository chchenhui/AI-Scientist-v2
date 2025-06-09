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

# Experiment data structure
experiment_data = {
    "adam_beta1": {
        "synthetic": {
            "beta1_list": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Hyperparameter sweep
betas = [0.5, 0.8, 0.9, 0.99]
epochs = 20

for beta in betas:
    print(f"Running Adam beta1={beta}")
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(beta, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    train_auc_list, val_auc_list = [], []
    train_loss_list, val_loss_list = [], []
    final_preds, final_labels = None, None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        t_losses, t_preds, t_labels = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
            t_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            t_labels.append(yb.detach().cpu().numpy())
        train_loss = np.mean(t_losses)
        train_auc = roc_auc_score(np.concatenate(t_labels), np.concatenate(t_preds))
        train_loss_list.append(train_loss)
        train_auc_list.append(train_auc)

        # Validation
        model.eval()
        v_losses, v_preds, v_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(1)
                loss = loss_fn(logits, yb)
                v_losses.append(loss.item())
                v_preds.append(torch.sigmoid(logits).cpu().numpy())
                v_labels.append(yb.cpu().numpy())
        val_loss = np.mean(v_losses)
        val_preds_full = np.concatenate(v_preds)
        val_labels_full = np.concatenate(v_labels)
        val_auc = roc_auc_score(val_labels_full, val_preds_full)
        val_loss_list.append(val_loss)
        val_auc_list.append(val_auc)

        print(f"  Epoch {epoch}: val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")
        if epoch == epochs:
            final_preds, final_labels = val_preds_full, val_labels_full

    # Record results
    sd = experiment_data["adam_beta1"]["synthetic"]
    sd["beta1_list"].append(beta)
    sd["metrics"]["train"].append(train_auc_list)
    sd["metrics"]["val"].append(val_auc_list)
    sd["losses"]["train"].append(train_loss_list)
    sd["losses"]["val"].append(val_loss_list)
    sd["predictions"].append(final_preds)
    sd["ground_truth"].append(final_labels)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
