import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
torch.manual_seed(0)
np.random.seed(0)

# synthetic data
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
    torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


# MLP with dropout
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# hyperparameters
dropout_rates = [0.0, 0.1, 0.2, 0.5]
num_epochs = 10

# initialize experiment data
experiment_data = {
    "dropout_rate": {
        "synthetic": {
            "hyperparams": dropout_rates,
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# training loop over dropout rates
for dr in dropout_rates:
    # models, optimizer, loss
    ai_model = MLP(D, 32, C, dr).to(device)
    user_model = MLP(D, 32, C, dr).to(device)
    optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
    optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # storage per epoch
    train_metrics, val_metrics = [], []
    train_losses, val_losses = [], []
    # epochs
    for epoch in range(1, num_epochs + 1):
        ai_model.train()
        user_model.train()
        train_loss_sum = 0.0
        train_jsd_sum = 0.0
        train_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # forward
            ai_logits = ai_model(x_batch)
            user_logits = user_model(x_batch)
            # losses
            loss_ai = loss_fn(ai_logits, y_batch)
            # backward ai
            optimizer_ai.zero_grad()
            loss_ai.backward()
            optimizer_ai.step()
            # backward user
            optimizer_user.zero_grad()
            loss_user = loss_fn(user_logits, y_batch)
            loss_user.backward()
            optimizer_user.step()
            # accumulate
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
        train_metrics.append(train_align)
        # validation
        ai_model.eval()
        user_model.eval()
        val_loss_sum = 0.0
        val_jsd_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                ai_logits = ai_model(x_batch)
                user_logits = user_model(x_batch)
                bs = y_batch.size(0)
                val_loss_sum += loss_fn(ai_logits, y_batch).item() * bs
                P = F.softmax(ai_logits, dim=1)
                Q = F.softmax(user_logits, dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                val_jsd_sum += torch.sum(1 - jsd).item()
                val_samples += bs
        val_loss = val_loss_sum / len(val_dataset)
        val_align = val_jsd_sum / val_samples
        val_losses.append(val_loss)
        val_metrics.append(val_align)
        print(f"dropout={dr} epoch={epoch} val_loss={val_loss:.4f}")
    # final predictions
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            preds = torch.argmax(ai_model(x_batch), dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(y_batch.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    # save run results
    exp = experiment_data["dropout_rate"]["synthetic"]
    exp["metrics"]["train"].append(train_metrics)
    exp["metrics"]["val"].append(val_metrics)
    exp["losses"]["train"].append(train_losses)
    exp["losses"]["val"].append(val_losses)
    exp["predictions"].append(all_preds)
    exp["ground_truth"].append(all_gts)

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
