import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
# Normalize features
mean = x_train.mean(axis=0)
std = x_train.std(axis=0) + 1e-8
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std

# Create DataLoaders
train_dataset = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


# Define simple MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


ai_model = MLP(D, 32, C).to(device)
user_model = MLP(D, 32, C).to(device)
optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Initialize experiment data
experiment_data = {
    "synthetic": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "epochs": [],
    }
}

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    ai_model.train()
    user_model.train()
    train_loss_sum = 0.0
    train_jsd_sum = 0.0
    train_samples = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # Forward
        ai_logits = ai_model(x_batch)
        user_logits = user_model(x_batch)
        # Losses and updates
        loss_ai = loss_fn(ai_logits, y_batch)
        loss_user = loss_fn(user_logits, y_batch)
        optimizer_ai.zero_grad()
        loss_ai.backward()
        optimizer_ai.step()
        optimizer_user.zero_grad()
        loss_user.backward()
        optimizer_user.step()
        bs = y_batch.size(0)
        train_loss_sum += loss_ai.item() * bs
        # Compute alignment (MMAS)
        P = F.softmax(ai_logits, dim=1)
        Q = F.softmax(user_logits, dim=1)
        M = 0.5 * (P + Q)
        kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
        kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
        jsd = 0.5 * (kl1 + kl2)
        train_jsd_sum += torch.sum(1 - jsd).item()
        train_samples += bs

    train_loss = train_loss_sum / len(train_dataset)
    train_align = train_jsd_sum / train_samples

    ai_model.eval()
    user_model.eval()
    val_loss_sum = 0.0
    val_jsd_sum = 0.0
    val_samples = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            ai_logits = ai_model(x_batch)
            bs = y_batch.size(0)
            val_loss_sum += loss_fn(ai_logits, y_batch).item() * bs
            P = F.softmax(ai_logits, dim=1)
            Q = F.softmax(user_model(x_batch), dim=1)
            M = 0.5 * (P + Q)
            kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
            kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
            jsd = 0.5 * (kl1 + kl2)
            val_jsd_sum += torch.sum(1 - jsd).item()
            val_samples += bs

    val_loss = val_loss_sum / len(val_dataset)
    val_align = val_jsd_sum / val_samples
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

    experiment_data["synthetic"]["metrics"]["train"].append(train_align)
    experiment_data["synthetic"]["metrics"]["val"].append(val_align)
    experiment_data["synthetic"]["losses"]["train"].append(train_loss)
    experiment_data["synthetic"]["losses"]["val"].append(val_loss)
    experiment_data["synthetic"]["epochs"].append(epoch)

# Final validation predictions and ground truth
all_preds, all_gts = [], []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        preds = torch.argmax(ai_model(x_batch), dim=1).cpu().numpy()
        all_preds.append(preds)
        all_gts.append(y_batch.numpy())
all_preds = np.concatenate(all_preds, axis=0)
all_gts = np.concatenate(all_gts, axis=0)
experiment_data["synthetic"]["predictions"] = all_preds
experiment_data["synthetic"]["ground_truth"] = all_gts

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
