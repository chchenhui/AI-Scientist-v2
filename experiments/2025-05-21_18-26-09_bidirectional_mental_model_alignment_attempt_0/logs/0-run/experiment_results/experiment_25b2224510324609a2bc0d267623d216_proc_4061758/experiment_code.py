import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
N_train, N_val = 1000, 200
max_features = 2000
batch_size = 32
num_epochs = 5
lr = 1e-3
dataset_names = ["ag_news", "imdb", "yelp_polarity"]


# MLP definition
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Utility: build TF-IDF
def build_vocab_idf(texts, max_features):
    from collections import Counter, defaultdict

    N = len(texts)
    df = defaultdict(int)
    tf_counts = Counter()
    for doc in texts:
        tokens = doc.lower().split()
        seen = set()
        for t in tokens:
            tf_counts[t] += 1
            if t not in seen:
                df[t] += 1
                seen.add(t)
    # top features
    top = [w for w, _ in tf_counts.most_common(max_features)]
    idf = np.array([np.log((N + 1) / (df[w] + 1)) + 1 for w in top], dtype=np.float32)
    vocab = {w: i for i, w in enumerate(top)}
    return vocab, idf


def texts_to_tfidf(texts, vocab, idf):
    X = np.zeros((len(texts), len(idf)), dtype=np.float32)
    for i, doc in enumerate(texts):
        for t in doc.lower().split():
            idx = vocab.get(t)
            if idx is not None:
                X[i, idx] += 1
    X *= idf[np.newaxis, :]
    return X


# Loss
loss_fn = nn.CrossEntropyLoss()

experiment_data = {}
for ds_name in dataset_names:
    # load and split
    raw = (
        load_dataset(ds_name, split="train")
        .shuffle(seed=42)
        .select(range(N_train + N_val))
    )
    texts = raw["text"]
    labels = np.array(raw["label"])
    train_texts, val_texts = texts[:N_train], texts[N_train:]
    y_train, y_val = labels[:N_train], labels[N_train:]
    C = int(labels.max()) + 1

    # build TF-IDF
    vocab, idf = build_vocab_idf(train_texts, max_features)
    X_train = texts_to_tfidf(train_texts, vocab, idf)
    X_val = texts_to_tfidf(val_texts, vocab, idf)
    # standardize
    mean = X_train.mean(0)
    std = X_train.std(0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # prepare experiment storage
    experiment_data[ds_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    data = experiment_data[ds_name]

    # models + optimizers
    ai_model = MLP(max_features, 128, C).to(device)
    user_model = MLP(max_features, 128, C).to(device)
    optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=lr)
    optimizer_user = torch.optim.Adam(user_model.parameters(), lr=lr)

    # training loop
    for epoch in range(1, num_epochs + 1):
        ai_model.train()
        user_model.train()
        total_loss = total_a1 = total_a2 = n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_ai = ai_model(xb)
            logits_user = user_model(xb)
            # losses
            loss_ai = loss_fn(logits_ai, yb)
            loss_user = loss_fn(logits_user, yb)
            optimizer_ai.zero_grad()
            loss_ai.backward()
            optimizer_ai.step()
            optimizer_user.zero_grad()
            loss_user.backward()
            optimizer_user.step()
            # alignments
            P = F.softmax(logits_ai, dim=1)
            Q = F.softmax(logits_user, dim=1)
            M = 0.5 * (P + Q)
            kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), 1)
            kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), 1)
            a1 = torch.mean(1 - 0.5 * (kl1 + kl2)).item()
            Pgt = torch.zeros_like(Q).scatter_(1, yb.unsqueeze(1), 1.0)
            M2 = 0.5 * (Q + Pgt)
            k1 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M2 + 1e-8)), 1)
            k2 = torch.sum(Pgt * (torch.log(Pgt + 1e-8) - torch.log(M2 + 1e-8)), 1)
            a2 = torch.mean(1 - 0.5 * (k1 + k2)).item()
            bs = yb.size(0)
            total_loss += loss_ai.item() * bs
            total_a1 += a1 * bs
            total_a2 += a2 * bs
            n_samples += bs

        train_loss = total_loss / len(train_ds)
        ta1 = total_a1 / n_samples
        ta2 = total_a2 / n_samples
        train_mai = 2 * (ta1 * ta2) / (ta1 + ta2 + 1e-8)
        data["losses"]["train"].append(train_loss)
        data["metrics"]["train"].append(train_mai)

        # validation
        ai_model.eval()
        user_model.eval()
        v_loss = v_a1 = v_a2 = v_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits_ai = ai_model(xb)
                v_loss += loss_fn(logits_ai, yb).item() * yb.size(0)
                P = F.softmax(logits_ai, 1)
                Q = F.softmax(user_model(xb), 1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), 1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), 1)
                v_a1 += torch.sum(1 - 0.5 * (kl1 + kl2)).item()
                Pgt = torch.zeros_like(Q).scatter_(1, yb.unsqueeze(1), 1.0)
                M2 = 0.5 * (Q + Pgt)
                k1 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M2 + 1e-8)), 1)
                k2 = torch.sum(Pgt * (torch.log(Pgt + 1e-8) - torch.log(M2 + 1e-8)), 1)
                v_a2 += torch.sum(1 - 0.5 * (k1 + k2)).item()
                v_samples += yb.size(0)

        val_loss = v_loss / len(val_ds)
        va1 = v_a1 / v_samples
        va2 = v_a2 / v_samples
        val_mai = 2 * (va1 * va2) / (va1 + va2 + 1e-8)
        data["losses"]["val"].append(val_loss)
        data["metrics"]["val"].append(val_mai)

        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        print(f"Dataset {ds_name} Epoch {epoch}: MAI = {val_mai:.4f}")

    # final predictions & ground truth
    preds, truth = [], []
    ai_model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = ai_model(xb)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            truth.extend(yb.cpu().numpy().tolist())
    data["predictions"] = preds
    data["ground_truth"] = truth

# Save all experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
