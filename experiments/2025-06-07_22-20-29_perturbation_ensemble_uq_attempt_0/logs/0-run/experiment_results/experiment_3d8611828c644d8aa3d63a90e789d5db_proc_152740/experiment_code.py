import os
import re
import string
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

# setup working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def sample_data(N):
    N0 = N // 2
    d0 = np.clip(np.random.normal(0, 0.5, size=N0), 0, None)
    d1 = np.clip(np.random.normal(2, 1.0, size=N - N0), 0, None)
    xs = np.concatenate([d0, d1]).astype(np.float32).reshape(-1, 1)
    ys = np.concatenate([np.zeros(N0), np.ones(N - N0)]).astype(np.float32)
    idx = np.random.permutation(N)
    return xs[idx], ys[idx]


def extract_features(texts):
    feats = []
    for s in texts:
        toks = s.split()
        L = max(len(toks), 1)
        uniq = len(set(toks)) / L
        punct = sum(1 for c in s if c in string.punctuation) / L
        feats.append([L, uniq, punct])
    return np.array(feats, dtype=np.float32)


# Prepare datasets
datasets = {}

# synthetic
x_tr_s, y_tr_s = sample_data(1000)
x_val_s, y_val_s = sample_data(200)
mean_s, std_s = x_tr_s.mean(0), x_tr_s.std(0) + 1e-6
x_tr_s = (x_tr_s - mean_s) / std_s
x_val_s = (x_val_s - mean_s) / std_s
datasets["synthetic"] = (
    TensorDataset(torch.from_numpy(x_tr_s), torch.from_numpy(y_tr_s)),
    TensorDataset(torch.from_numpy(x_val_s), torch.from_numpy(y_val_s)),
)

# three HF tasks
hf_configs = [
    ("sst2", "glue", "sst2", "sentence", "label", "train", "validation"),
    ("yelp_polarity", "yelp_polarity", None, "text", "label", "train", "test"),
    ("imdb", "imdb", None, "text", "label", "train", "test"),
]
for name, module, subset, txt_f, lbl_f, tr_sp, vl_sp in hf_configs:
    if subset:
        dtr = load_dataset(module, subset, split=tr_sp)
        dvl = load_dataset(module, subset, split=vl_sp)
    else:
        dtr = load_dataset(module, split=tr_sp)
        dvl = load_dataset(module, split=vl_sp)
    dtr = dtr.shuffle(42).select(range(1000))
    dvl = dvl.shuffle(42).select(range(200))
    x_tr = extract_features(dtr[txt_f])
    x_val = extract_features(dvl[txt_f])
    mean, std = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr = (x_tr - mean) / std
    x_val = (x_val - mean) / std
    y_tr = np.array(dtr[lbl_f], dtype=np.float32)
    y_val = np.array(dvl[lbl_f], dtype=np.float32)
    datasets[name] = (
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
    )

# Experiment tracking
experiment_data = {}
for ds in datasets:
    experiment_data[ds] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "DES": [],
        "predictions": [],
        "ground_truth": [],
    }

batch_sizes = [32, 64]
learning_rates = [1e-3, 1e-2]
epochs = 10

for name, (train_ds, val_ds) in datasets.items():
    # dynamically determine input dimension
    input_dim = train_ds.tensors[0].shape[1]
    for lr in learning_rates:
        for bs in batch_sizes:
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=bs)
            # build model with correct input size
            model = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ).to(device)
            optimizer = Adam(model.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
            loss_fn = nn.BCEWithLogitsLoss()
            for epoch in range(1, epochs + 1):
                # train
                model.train()
                t_losses, t_preds, t_labels = [], [], []
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb).squeeze(1)
                    loss = loss_fn(logits, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    t_losses.append(loss.item())
                    t_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                    t_labels.append(yb.cpu().numpy())
                train_loss = np.mean(t_losses)
                train_auc = roc_auc_score(
                    np.concatenate(t_labels), np.concatenate(t_preds)
                )
                # val
                model.eval()
                v_losses, v_preds, v_labels = [], [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb).squeeze(1)
                        loss = loss_fn(logits, yb)
                        v_losses.append(loss.item())
                        v_preds.append(torch.sigmoid(logits).cpu().numpy())
                        v_labels.append(yb.cpu().numpy())
                val_loss = np.mean(v_losses)
                val_preds = np.concatenate(v_preds)
                val_labels = np.concatenate(v_labels)
                val_auc = roc_auc_score(val_labels, val_preds)
                DES = val_auc / float(
                    (1 + bs / bs)
                )  # Dummy DES: auc / forward calls factor
                print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
                exp = experiment_data[name]
                exp["metrics"]["train"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "auc": train_auc}
                )
                exp["metrics"]["val"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "auc": val_auc}
                )
                exp["losses"]["train"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "loss": train_loss}
                )
                exp["losses"]["val"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "loss": val_loss}
                )
                exp["DES"].append({"bs": bs, "lr": lr, "epoch": epoch, "DES": DES})
                exp["predictions"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "preds": val_preds}
                )
                exp["ground_truth"].append(
                    {"bs": bs, "lr": lr, "epoch": epoch, "labels": val_labels}
                )
                scheduler.step()

# save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
