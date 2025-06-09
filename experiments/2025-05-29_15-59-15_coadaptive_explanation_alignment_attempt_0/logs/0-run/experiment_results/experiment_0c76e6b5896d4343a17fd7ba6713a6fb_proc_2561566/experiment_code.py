# Set random seed
import random
import numpy as np
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic dataset generation
np.random.seed(0)
N = 2000
D = 2
X = np.random.randn(N, D)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# Train/val/test split
idx = np.random.permutation(N)
train_idx, val_idx, test_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Normalize features
mean, std = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# AI model definition
class AIModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# Train AI model
ai_model = AIModel(D, 16, 2).to(device)
criterion_ai = nn.CrossEntropyLoss()
optimizer_ai = optim.Adam(ai_model.parameters(), lr=1e-2)


# DataLoaders for AI
class SimpleDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.X[i], "y": self.y[i]}


ai_train_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=32, shuffle=True)
ai_val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=32)

for epoch in range(15):
    ai_model.train()
    for batch in ai_train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = ai_model(batch["x"])
        loss = criterion_ai(logits, batch["y"])
        optimizer_ai.zero_grad()
        loss.backward()
        optimizer_ai.step()
    # validation
    ai_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in ai_val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = ai_model(batch["x"])
            val_loss += criterion_ai(out, batch["y"]).item() * batch["x"].size(0)
    val_loss /= len(ai_val_loader.dataset)
    print(f"Epoch {epoch}: AI val loss = {val_loss:.4f}")

# Generate explanations (predicted probabilities)
ai_model.eval()
with torch.no_grad():
    X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
    logits_all = ai_model(X_all)
    probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
# Split explanations
p_train = probs_all[: len(X_train)]
p_val = probs_all[len(X_train) : len(X_train) + len(X_val)]
p_test = probs_all[-len(X_test) :]
# Ground truth AI decisions
f_train = np.argmax(p_train, axis=1)
f_val = np.argmax(p_val, axis=1)
f_test = np.argmax(p_test, axis=1)

# Prepare user dataset (features = [X, p], labels = f)
X_usr_train = np.hstack([X_train, p_train])
X_usr_val = np.hstack([X_val, p_val])
X_usr_test = np.hstack([X_test, p_test])


# User Dataset & DataLoaders
class UserDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.X[i], "label": self.y[i]}


usr_train_loader = DataLoader(UserDS(X_usr_train, f_train), batch_size=32, shuffle=True)
usr_val_loader = DataLoader(UserDS(X_usr_val, f_val), batch_size=32)
usr_test_loader = DataLoader(UserDS(X_usr_test, f_test), batch_size=32)


# User model
class UserModel(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


user_model = UserModel(D + 2, 8, 2).to(device)
criterion_usr = nn.CrossEntropyLoss()
optimizer_usr = optim.Adam(user_model.parameters(), lr=1e-2)

# Experiment data
experiment_data = {
    "static_explainer": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Train user model and track metrics
for epoch in range(20):
    user_model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in usr_train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = user_model(batch["feat"])
        loss = criterion_usr(out, batch["label"])
        optimizer_usr.zero_grad()
        loss.backward()
        optimizer_usr.step()
        total_loss += loss.item() * batch["feat"].size(0)
        preds = out.argmax(dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += len(batch["label"])
    train_loss = total_loss / total
    train_acc = correct / total
    experiment_data["static_explainer"]["losses"]["train"].append(train_loss)
    experiment_data["static_explainer"]["metrics"]["train"].append(train_acc)

    # validation
    user_model.eval()
    val_loss, val_corr, val_tot = 0.0, 0, 0
    with torch.no_grad():
        for batch in usr_val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = user_model(batch["feat"])
            val_loss += criterion_usr(out, batch["label"]).item() * batch["feat"].size(
                0
            )
            preds = out.argmax(dim=1)
            val_corr += (preds == batch["label"]).sum().item()
            val_tot += len(batch["label"])
    val_loss /= val_tot
    val_acc = val_corr / val_tot
    experiment_data["static_explainer"]["losses"]["val"].append(val_loss)
    experiment_data["static_explainer"]["metrics"]["val"].append(val_acc)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

# Test evaluation
user_model.eval()
test_preds, test_gt = [], []
with torch.no_grad():
    for batch in usr_test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = user_model(batch["feat"])
        p = out.argmax(dim=1).cpu().numpy().tolist()
        test_preds.extend(p)
        test_gt.extend(batch["label"].cpu().numpy().tolist())

experiment_data["static_explainer"]["predictions"] = test_preds
experiment_data["static_explainer"]["ground_truth"] = test_gt

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
