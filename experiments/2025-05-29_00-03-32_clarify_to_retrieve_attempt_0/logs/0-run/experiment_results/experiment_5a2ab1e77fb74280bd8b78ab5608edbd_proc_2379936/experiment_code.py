import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def make_xor(n):
    X = torch.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).long()
    return X, y


# Prepare datasets
train_X, train_y = make_xor(2000)
val_X, val_y = make_xor(500)
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=64)


# Model definition
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


# Vectorized compute of accuracy gain per clarification
def compute_accuracy_gain(loader, model, threshold, mc_T):
    model.eval()
    base_corr = clar_corr = clar_count = total_N = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        # mask ambiguous feature
        X_mask = Xb.clone()
        X_mask[:, 1] = 0
        # base prediction
        with torch.no_grad():
            out_base = model(X_mask)
            preds_base = out_base.argmax(dim=1)
        base_corr += (preds_base == yb).sum().item()
        # MC dropout sampling
        model.train()
        ps = []
        with torch.no_grad():
            for _ in range(mc_T):
                p = torch.softmax(model(X_mask), dim=1)
                ps.append(p)
        ps = torch.stack(ps, dim=0)  # [T, batch, classes]
        var = ps.var(dim=0).sum(dim=1)  # [batch]
        clar_mask = var > threshold
        clar_count += clar_mask.sum().item()
        # clarified prediction
        model.eval()
        with torch.no_grad():
            out_clar = model(Xb)
            preds_clar = out_clar.argmax(dim=1)
        # combine correctness
        correct = torch.where(clar_mask, preds_clar == yb, preds_base == yb)
        clar_corr += correct.sum().item()
        total_N += Xb.size(0)
    base_acc = base_corr / total_N
    clar_acc = clar_corr / total_N
    avg_ct = clar_count / total_N if total_N > 0 else 0
    return (clar_acc - base_acc) / avg_ct if avg_ct > 0 else 0.0


# Hyperparameters
weight_decays = [0, 1e-5, 1e-4, 1e-3]
epochs = 10
threshold = 0.02
mc_T = 5

# Data container
experiment_data = {
    "accuracy_gain_per_clarification": {"train": [], "val": []},
    "losses": {"train": [], "val": []},
}

# Main loop
for wd in weight_decays:
    print(f"Running weight_decay = {wd}")
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    tr_metrics, val_metrics = [], []
    tr_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        tr_losses.append(train_loss)

        # Compute accuracy gain on train
        ag_train = compute_accuracy_gain(train_loader, model, threshold, mc_T)
        tr_metrics.append(ag_train)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                total_val_loss += criterion(out, yb).item() * Xb.size(0)
        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Compute accuracy gain on val
        ag_val = compute_accuracy_gain(val_loader, model, threshold, mc_T)
        val_metrics.append(ag_val)

        # Print metrics
        print(
            f"Epoch {epoch}: validation_loss = {val_loss:.4f}, "
            + f"accuracy_gain_per_clarification_train = {ag_train:.4f}, "
            + f"accuracy_gain_per_clarification_val = {ag_val:.4f}"
        )

    experiment_data["accuracy_gain_per_clarification"]["train"].append(tr_metrics)
    experiment_data["accuracy_gain_per_clarification"]["val"].append(val_metrics)
    experiment_data["losses"]["train"].append(tr_losses)
    experiment_data["losses"]["val"].append(val_losses)

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
