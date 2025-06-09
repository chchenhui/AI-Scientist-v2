import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Synthetic dataset
N = 1000
X = np.random.randn(N, 2)
w_true = np.array([1.0, 1.0])
y = (X.dot(w_true) + 0.5 * np.random.randn(N) > 0).astype(int)
split = int(0.8 * N)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# Convert to torch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)


# Model definition
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Simulated human mental model bias
w_human = torch.tensor([0.5, -0.5], device=device)

# Initialize experiment data
experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Training loop
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    # Training
    model.train()
    total_train_loss = 0.0
    for x, yb in train_loader:
        x, yb = x.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * x.size(0)
    train_loss = total_train_loss / len(train_loader.dataset)

    # Compute training alignment accuracy
    model.eval()
    correct_align_train = 0
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            logits = model(x)
            model_pred = logits.argmax(dim=1)
            human_pred = (x @ w_human > 0).long()
            correct_align_train += (human_pred == model_pred).sum().item()
    train_alignment = correct_align_train / len(train_loader.dataset)

    # Validation
    total_val_loss = 0.0
    correct_align_val = 0
    val_preds = []
    val_truth = []
    with torch.no_grad():
        for x, yb in val_loader:
            x, yb = x.to(device), yb.to(device)
            logits = model(x)
            loss = criterion(logits, yb)
            total_val_loss += loss.item() * x.size(0)
            model_pred = logits.argmax(dim=1)
            human_pred = (x @ w_human > 0).long()
            correct_align_val += (human_pred == model_pred).sum().item()
            val_preds.extend(model_pred.cpu().numpy().tolist())
            val_truth.extend(yb.cpu().numpy().tolist())
    val_loss = total_val_loss / len(val_loader.dataset)
    val_alignment = correct_align_val / len(val_loader.dataset)

    # Logging
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
    experiment_data["synthetic"]["losses"]["train"].append(train_loss)
    experiment_data["synthetic"]["losses"]["val"].append(val_loss)
    experiment_data["synthetic"]["metrics"]["train"].append(train_alignment)
    experiment_data["synthetic"]["metrics"]["val"].append(val_alignment)
    experiment_data["synthetic"]["predictions"].append(np.array(val_preds))
    experiment_data["synthetic"]["ground_truth"].append(np.array(val_truth))

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
