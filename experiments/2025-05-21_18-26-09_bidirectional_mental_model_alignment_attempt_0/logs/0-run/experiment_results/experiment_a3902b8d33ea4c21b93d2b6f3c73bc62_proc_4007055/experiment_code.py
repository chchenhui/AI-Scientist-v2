import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# set up working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device and reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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

# dataloaders
train_ds = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
)
val_ds = TensorDataset(
    torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


# model definition
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
weight_decays = [1e-5, 1e-4, 1e-3, 1e-2]

# prepare experiment_data
experiment_data = {
    "weight_decay": {
        "synthetic": {
            "weight_decay_values": [],
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_val.copy(),
        }
    }
}

# hyperparameter sweep
for wd in weight_decays:
    print(f"=== Training with weight_decay={wd} ===")
    experiment_data["weight_decay"]["synthetic"]["weight_decay_values"].append(wd)
    ai_model = MLP(D, 32, C).to(device)
    user_model = MLP(D, 32, C).to(device)
    optim_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3, weight_decay=wd)
    optim_usr = torch.optim.Adam(user_model.parameters(), lr=1e-3, weight_decay=wd)

    tr_metrics, vl_metrics = [], []
    tr_losses, vl_losses = [], []

    for epoch in range(1, num_epochs + 1):
        ai_model.train()
        user_model.train()
        train_loss_sum = train_jsd_sum = train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_ai = ai_model(xb)
            logits_usr = user_model(xb)
            loss_ai = loss_fn(logits_ai, yb)
            loss_usr = loss_fn(logits_usr, yb)
            optim_ai.zero_grad()
            loss_ai.backward()
            optim_ai.step()
            optim_usr.zero_grad()
            loss_usr.backward()
            optim_usr.step()
            bs = yb.size(0)
            train_loss_sum += loss_ai.item() * bs
            P = F.softmax(logits_ai, dim=1)
            Q = F.softmax(logits_usr, dim=1)
            M = 0.5 * (P + Q)
            kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
            kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
            jsd = 0.5 * (kl1 + kl2)
            train_jsd_sum += (1 - jsd).sum().item()
            train_n += bs
        tr_loss = train_loss_sum / len(train_ds)
        tr_align = train_jsd_sum / train_n

        ai_model.eval()
        user_model.eval()
        val_loss_sum = val_jsd_sum = val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits_ai = ai_model(xb)
                bs = yb.size(0)
                val_loss_sum += loss_fn(logits_ai, yb).item() * bs
                P = F.softmax(logits_ai, dim=1)
                Q = F.softmax(user_model(xb), dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                val_jsd_sum += (1 - jsd).sum().item()
                val_n += bs
        vl_loss = val_loss_sum / len(val_ds)
        vl_align = val_jsd_sum / val_n
        print(f"wd={wd} epoch={epoch} val_loss={vl_loss:.4f}")

        tr_metrics.append(tr_align)
        vl_metrics.append(vl_align)
        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)

    # store results
    experiment_data["weight_decay"]["synthetic"]["metrics"]["train"].append(tr_metrics)
    experiment_data["weight_decay"]["synthetic"]["metrics"]["val"].append(vl_metrics)
    experiment_data["weight_decay"]["synthetic"]["losses"]["train"].append(tr_losses)
    experiment_data["weight_decay"]["synthetic"]["losses"]["val"].append(vl_losses)

    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            p = torch.argmax(ai_model(xb), dim=1).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)
    experiment_data["weight_decay"]["synthetic"]["predictions"].append(preds)

# convert lists to np arrays for plotting
sd = experiment_data["weight_decay"]["synthetic"]
sd["metrics"]["train"] = np.array(sd["metrics"]["train"])
sd["metrics"]["val"] = np.array(sd["metrics"]["val"])
sd["losses"]["train"] = np.array(sd["losses"]["train"])
sd["losses"]["val"] = np.array(sd["losses"]["val"])
sd["predictions"] = np.stack(sd["predictions"], axis=0)
sd["ground_truth"] = np.array(sd["ground_truth"])
sd["weight_decay_values"] = np.array(sd["weight_decay_values"])

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
