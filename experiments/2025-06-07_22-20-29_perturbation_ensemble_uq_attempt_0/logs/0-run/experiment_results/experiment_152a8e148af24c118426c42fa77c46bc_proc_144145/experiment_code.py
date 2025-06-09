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


# Prepare datasets once
x_train, y_train = sample_data(1000)
x_val, y_val = sample_data(200)
train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Hyperparameter sweep for beta2 values
beta2_list = [0.9, 0.95, 0.99, 0.999, 0.9999]
experiment_data = {
    "adam_beta2": {
        "synthetic": {
            "beta2_values": beta2_list,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Training settings
epochs = 20
lr = 0.01
beta1 = 0.9

# Sweep
for beta2 in beta2_list:
    # New model & optimizer for each beta2
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    loss_fn = nn.BCEWithLogitsLoss()

    # Per-epoch records
    train_auc_ep, val_auc_ep = [], []
    train_loss_ep, val_loss_ep = [], []
    preds_ep, gt_ep = [], []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        losses_tr, preds_tr, labels_tr = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_tr.append(loss.item())
            preds_tr.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels_tr.append(yb.cpu().numpy())
        train_loss = np.mean(losses_tr)
        train_auc = roc_auc_score(np.concatenate(labels_tr), np.concatenate(preds_tr))
        train_loss_ep.append(train_loss)
        train_auc_ep.append(train_auc)

        # Validate
        model.eval()
        losses_val, preds_val, labels_val = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(1)
                loss = loss_fn(logits, yb)
                losses_val.append(loss.item())
                preds_val.append(torch.sigmoid(logits).cpu().numpy())
                labels_val.append(yb.cpu().numpy())
        val_loss = np.mean(losses_val)
        val_auc = roc_auc_score(np.concatenate(labels_val), np.concatenate(preds_val))
        val_loss_ep.append(val_loss)
        val_auc_ep.append(val_auc)

        preds_ep.append(np.concatenate(preds_val))
        gt_ep.append(np.concatenate(labels_val))

        print(
            f"beta2={beta2} epoch={epoch} val_loss={val_loss:.4f} val_auc={val_auc:.4f}"
        )

    # Store results for this beta2
    sd = experiment_data["adam_beta2"]["synthetic"]
    sd["metrics"]["train"].append(train_auc_ep)
    sd["metrics"]["val"].append(val_auc_ep)
    sd["losses"]["train"].append(train_loss_ep)
    sd["losses"]["val"].append(val_loss_ep)
    sd["predictions"].append(preds_ep)
    sd["ground_truth"].append(gt_ep)

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
