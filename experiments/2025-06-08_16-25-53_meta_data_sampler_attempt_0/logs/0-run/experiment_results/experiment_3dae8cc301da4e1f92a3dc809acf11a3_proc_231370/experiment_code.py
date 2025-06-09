import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Create synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)
X = X.astype(np.float32)
y = y.astype(np.int64)
# Split and normalize
idx = np.random.RandomState(42).permutation(len(X))
train_idx, val_idx = idx[:800], idx[800:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


# Simple MLP classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)

    def embed(self, x):
        return self.relu(self.fc1(x))


model = Classifier().to(device)
criterion_batch = nn.CrossEntropyLoss(reduction="none")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

features_list, contributions_list = [], []
train_loss_total, num_batches = 0.0, 0

# One epoch training with contribution measurement
for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    # validation loss before update
    model.eval()
    val_loss_before = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss_before += criterion(model(xb), yb).item()
    val_loss_before /= len(val_loader)
    # compute features on this batch
    model.train()
    out = model(x_batch)
    per_losses = criterion_batch(out, y_batch)
    avg_loss = per_losses.mean().item()
    embeds = model.embed(x_batch)
    mean_emb = embeds.mean(dim=0, keepdim=True)
    avg_div = torch.norm(embeds - mean_emb, dim=1).mean().item()
    features_list.append([avg_loss, avg_div])
    # update classifier
    loss_train = per_losses.mean()
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    train_loss_total += loss_train.item()
    # validation loss after update
    model.eval()
    val_loss_after = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss_after += criterion(model(xb), yb).item()
    val_loss_after /= len(val_loader)
    contributions_list.append(val_loss_before - val_loss_after)
    num_batches += 1

# Record classifier losses
train_loss_avg = train_loss_total / num_batches
model.eval()
final_val_loss = 0.0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        final_val_loss += criterion(model(xb), yb).item()
final_val_loss /= len(val_loader)
experiment_data["synthetic"]["losses"]["train"].append(train_loss_avg)
experiment_data["synthetic"]["losses"]["val"].append(final_val_loss)
print(f"Epoch 0: validation_loss = {final_val_loss:.4f}")

# Build DVN dataset
X_feat = torch.tensor(features_list, dtype=torch.float32)
y_true = torch.tensor(contributions_list, dtype=torch.float32)
dvn_loader = DataLoader(TensorDataset(X_feat, y_true), batch_size=16, shuffle=True)


# Simple DVN regressor
class DVN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


dvn = DVN().to(device)
dvn_optimizer = optim.Adam(dvn.parameters(), lr=1e-3)
mse = nn.MSELoss()

# Train DVN and track Spearman correlation
for epoch in range(20):
    dvn.train()
    for xb, yb in dvn_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = dvn(xb).squeeze()
        loss_dvn = mse(pred, yb)
        dvn_optimizer.zero_grad()
        loss_dvn.backward()
        dvn_optimizer.step()
    dvn.eval()
    with torch.no_grad():
        y_pred = dvn(X_feat.to(device)).cpu().squeeze().numpy()
        y_gt = y_true.numpy()
    corr = spearmanr(y_gt, y_pred).correlation
    experiment_data["synthetic"]["metrics"]["train"].append(corr)

# Save predictions, ground truth, and all metrics
experiment_data["synthetic"]["predictions"] = y_pred.tolist()
experiment_data["synthetic"]["ground_truth"] = y_gt.tolist()
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Contribution Prediction Correlation: {corr:.4f}")
