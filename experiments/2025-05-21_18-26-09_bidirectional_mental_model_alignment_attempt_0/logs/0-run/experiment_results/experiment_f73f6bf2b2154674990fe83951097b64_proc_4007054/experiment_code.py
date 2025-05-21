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
mean, std = x_train.mean(0), x_train.std(0) + 1e-8
x_train = (x_train - mean) / std
x_val = (x_val - mean) / std

train_dataset = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
)


# simple MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


loss_fn = nn.CrossEntropyLoss()
num_epochs = 10
batch_sizes = [16, 32, 128, 256]

# initialize experiment data container
experiment_data = {
    "batch_size": {
        "synthetic": {
            "batch_sizes": [],
            "epochs": list(range(1, num_epochs + 1)),
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_val,
        }
    }
}

for bs in batch_sizes:
    print(f"Starting run with batch size = {bs}")
    experiment_data["batch_size"]["synthetic"]["batch_sizes"].append(bs)
    # re-seed for reproducibility across runs
    torch.manual_seed(0)
    np.random.seed(0)
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs)
    # fresh models and optimizers
    ai_model = MLP(D, 32, C).to(device)
    user_model = MLP(D, 32, C).to(device)
    optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
    optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)

    train_alignments, val_alignments = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        ai_model.train()
        user_model.train()
        t_loss_sum = 0.0
        t_jsd_sum = 0.0
        t_samples = 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            ai_logits = ai_model(x_b)
            user_logits = user_model(x_b)
            loss_ai = loss_fn(ai_logits, y_b)
            loss_user = loss_fn(user_logits, y_b)

            optimizer_ai.zero_grad()
            loss_ai.backward()
            optimizer_ai.step()
            optimizer_user.zero_grad()
            loss_user.backward()
            optimizer_user.step()

            bs_cur = y_b.size(0)
            t_loss_sum += loss_ai.item() * bs_cur
            P = F.softmax(ai_logits, dim=1)
            Q = F.softmax(user_logits, dim=1)
            M = 0.5 * (P + Q)
            kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
            kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
            jsd = 0.5 * (kl1 + kl2)
            t_jsd_sum += torch.sum(1 - jsd).item()
            t_samples += bs_cur

        train_loss = t_loss_sum / len(train_dataset)
        train_align = t_jsd_sum / t_samples

        # validation
        ai_model.eval()
        user_model.eval()
        v_loss_sum = 0.0
        v_jsd_sum = 0.0
        v_samples = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                ai_logits = ai_model(x_b)
                bs_cur = y_b.size(0)
                v_loss_sum += loss_fn(ai_logits, y_b).item() * bs_cur
                P = F.softmax(ai_logits, dim=1)
                Q = F.softmax(user_model(x_b), dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                v_jsd_sum += torch.sum(1 - jsd).item()
                v_samples += bs_cur

        val_loss = v_loss_sum / len(val_dataset)
        val_align = v_jsd_sum / v_samples
        print(
            f"[bs={bs}] Epoch {epoch}: val_loss={val_loss:.4f}, val_align={val_align:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_alignments.append(train_align)
        val_alignments.append(val_align)

    # record curves
    experiment_data["batch_size"]["synthetic"]["metrics"]["train"].append(
        train_alignments
    )
    experiment_data["batch_size"]["synthetic"]["metrics"]["val"].append(val_alignments)
    experiment_data["batch_size"]["synthetic"]["losses"]["train"].append(train_losses)
    experiment_data["batch_size"]["synthetic"]["losses"]["val"].append(val_losses)

    # final predictions
    preds = []
    with torch.no_grad():
        for x_b, _ in val_loader:
            x_b = x_b.to(device)
            p = torch.argmax(ai_model(x_b), dim=1).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)
    experiment_data["batch_size"]["synthetic"]["predictions"].append(preds)

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
