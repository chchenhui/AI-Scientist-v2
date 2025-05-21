import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

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


# Hyperparameter sweep over init_std
init_stds = [0.01, 0.1, 0.5, 1.0]
num_epochs = 10
experiment_data = {"init_std": {}}

for init_std in init_stds:
    # Prepare storage
    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "epochs": [],
        "predictions": [],
        "ground_truth": [],
    }
    # Instantiate and initialize models
    ai_model = MLP(D, 32, C).to(device)
    user_model = MLP(D, 32, C).to(device)
    for model in (ai_model, user_model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
                nn.init.zeros_(m.bias)
    optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
    optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        ai_model.train()
        user_model.train()
        train_loss_sum, train_jsd_sum, train_samples = 0.0, 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Forward & backward for AI model
            ai_logits = ai_model(x_batch)
            loss_ai = loss_fn(ai_logits, y_batch)
            optimizer_ai.zero_grad()
            loss_ai.backward()
            optimizer_ai.step()
            # Forward & backward for User model
            user_logits = user_model(x_batch)
            loss_user = loss_fn(user_logits, y_batch)
            optimizer_user.zero_grad()
            loss_user.backward()
            optimizer_user.step()
            # Accumulate loss and alignment
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

        # Validation
        ai_model.eval()
        user_model.eval()
        val_loss_sum, val_jsd_sum, val_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                ai_logits = ai_model(x_batch)
                loss_val = loss_fn(ai_logits, y_batch)
                bs = y_batch.size(0)
                val_loss_sum += loss_val.item() * bs
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

        # Record
        exp["metrics"]["train"].append(train_align)
        exp["metrics"]["val"].append(val_align)
        exp["losses"]["train"].append(train_loss)
        exp["losses"]["val"].append(val_loss)
        exp["epochs"].append(epoch)
        print(f"init_std={init_std} epoch={epoch} val_loss={val_loss:.4f}")

    # Final validation predictions
    all_preds, all_gts = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            preds = torch.argmax(ai_model(x_batch), dim=1).cpu().numpy()
            all_preds.append(preds)
            all_gts.append(y_batch.numpy())
    exp["predictions"] = np.concatenate(all_preds, axis=0)
    exp["ground_truth"] = np.concatenate(all_gts, axis=0)

    experiment_data["init_std"][str(init_std)] = exp

# Convert all lists to numpy arrays
for std_key, exp in experiment_data["init_std"].items():
    exp["metrics"]["train"] = np.array(exp["metrics"]["train"])
    exp["metrics"]["val"] = np.array(exp["metrics"]["val"])
    exp["losses"]["train"] = np.array(exp["losses"]["train"])
    exp["losses"]["val"] = np.array(exp["losses"]["val"])
    exp["epochs"] = np.array(exp["epochs"])
    exp["predictions"] = np.array(exp["predictions"])
    exp["ground_truth"] = np.array(exp["ground_truth"])

# Save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
