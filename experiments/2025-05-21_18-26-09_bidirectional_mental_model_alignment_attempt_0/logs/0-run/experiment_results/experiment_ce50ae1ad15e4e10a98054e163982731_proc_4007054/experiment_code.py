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

# Hyperparameters
alphas = [0.0, 0.05, 0.1, 0.2]
num_epochs = 10
batch_size = 64
lr = 1e-3

# Reproducibility for data
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

# TensorDatasets & Loaders (shuffle will be controlled by the seed)
train_dataset = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Simple MLP definition
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Prepare experiment_data
experiment_data = {
    "label_smoothing": {
        "synthetic": {
            "alphas": alphas,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "epochs": list(range(1, num_epochs + 1)),
        }
    }
}

# Hyperparameter sweep
for alpha in alphas:
    # reset seeds for reproducibility of weights & shuffles
    torch.manual_seed(0)
    np.random.seed(0)
    # models & optimizers
    ai_model = MLP(D, 32, C).to(device)
    user_model = MLP(D, 32, C).to(device)
    optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=lr)
    optimizer_user = torch.optim.Adam(user_model.parameters(), lr=lr)
    # loss with smoothing
    loss_fn = nn.CrossEntropyLoss(label_smoothing=alpha)

    train_aligns, val_aligns = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        ai_model.train()
        user_model.train()
        train_loss_sum = 0.0
        train_jsd_sum = 0.0
        train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            ai_logits = ai_model(x_batch)
            user_logits = user_model(x_batch)

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
        train_losses.append(train_loss)
        train_aligns.append(train_align)

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
        val_losses.append(val_loss)
        val_aligns.append(val_align)

        print(
            f"alpha={alpha} | Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

    # final predictions & ground truth
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            preds = torch.argmax(ai_model(x_batch), dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(y_batch.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    sd = experiment_data["label_smoothing"]["synthetic"]
    sd["metrics"]["train"].append(train_aligns)
    sd["metrics"]["val"].append(val_aligns)
    sd["losses"]["train"].append(train_losses)
    sd["losses"]["val"].append(val_losses)
    sd["predictions"].append(all_preds)
    sd["ground_truth"].append(all_gts)

# Save experiment_data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
