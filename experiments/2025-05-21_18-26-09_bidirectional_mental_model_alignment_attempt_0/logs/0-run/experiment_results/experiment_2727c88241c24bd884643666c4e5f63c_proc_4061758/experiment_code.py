import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)
np.random.seed(0)

# Load tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
bert.eval()
for p in bert.parameters():
    p.requires_grad = False


# Simple MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


dataset_names = ["ag_news", "dbpedia_14", "yelp_polarity"]
N_train, N_val = 500, 100
experiment_data = {}

for dataset_name in dataset_names:
    ds = load_dataset(dataset_name)
    ds_train = ds["train"].select(range(N_train))
    ds_val = ds["test"].select(range(N_val))
    # detect text column
    text_col = [
        c
        for c, f in ds_train.features.items()
        if getattr(f, "dtype", None) == "string" and c != "label"
    ][0]
    train_texts = ds_train[text_col]
    train_labels = np.array(ds_train["label"])
    val_texts = ds_val[text_col]
    val_labels = np.array(ds_val["label"])

    # embed extraction
    def extract(texts):
        embs = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            toks = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = bert(**toks)
                embs.append(out.last_hidden_state.mean(dim=1).cpu())
        return torch.cat(embs, dim=0)

    train_emb = extract(train_texts)
    val_emb = extract(val_texts)
    # normalize
    mean = train_emb.mean(0, keepdim=True)
    std = train_emb.std(0, keepdim=True) + 1e-8
    train_emb = (train_emb - mean) / std
    val_emb = (val_emb - mean) / std
    # DataLoaders
    train_ds = TensorDataset(train_emb, torch.tensor(train_labels, dtype=torch.long))
    val_ds = TensorDataset(val_emb, torch.tensor(val_labels, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    # models & optimizers
    num_labels = ds_train.features["label"].num_classes
    ai_model = MLP(train_emb.size(1), 128, num_labels).to(device)
    user_model = MLP(train_emb.size(1), 128, num_labels).to(device)
    optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
    optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # metrics storage
    train_losses, val_losses = [], []
    train_s1, train_s2, train_mai = [], [], []
    val_s1, val_s2, val_mai = [], [], []
    # training loop
    for epoch in range(1, 6):
        ai_model.train()
        user_model.train()
        tot_loss = tot1 = tot2 = cnt = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_ai = ai_model(xb)
            logits_user = user_model(xb)
            loss_ai = loss_fn(logits_ai, yb)
            loss_user = loss_fn(logits_user, yb)
            optimizer_ai.zero_grad()
            loss_ai.backward()
            optimizer_ai.step()
            optimizer_user.zero_grad()
            loss_user.backward()
            optimizer_user.step()
            bs = yb.size(0)
            tot_loss += loss_ai.item() * bs
            P = F.softmax(logits_ai, 1)
            Q = F.softmax(logits_user, 1)
            M = 0.5 * (P + Q)
            kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), 1)
            kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), 1)
            s1 = (1 - 0.5 * (kl1 + kl2)).sum().item()
            tot1 += s1
            Y = F.one_hot(yb, num_classes=num_labels).float()
            M2 = 0.5 * (Q + Y)
            klq = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M2 + 1e-8)), 1)
            kly = torch.sum(Y * (torch.log(Y + 1e-8) - torch.log(M2 + 1e-8)), 1)
            s2 = (1 - 0.5 * (klq + kly)).sum().item()
            tot2 += s2
            cnt += bs
        t_loss = tot_loss / len(train_ds)
        s1_avg = tot1 / cnt
        s2_avg = tot2 / cnt
        mai_t = 2 * s1_avg * s2_avg / (s1_avg + s2_avg + 1e-8)
        train_losses.append(t_loss)
        train_s1.append(s1_avg)
        train_s2.append(s2_avg)
        train_mai.append(mai_t)
        # validation
        ai_model.eval()
        user_model.eval()
        v_loss = v1 = v2 = vc = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                la = ai_model(xb)
                lu = user_model(xb)
                v_loss += loss_fn(la, yb).item() * yb.size(0)
                P = F.softmax(la, 1)
                Q = F.softmax(lu, 1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), 1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), 1)
                v1 += (1 - 0.5 * (kl1 + kl2)).sum().item()
                Y = F.one_hot(yb, num_classes=num_labels).float()
                M2 = 0.5 * (Q + Y)
                klq = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M2 + 1e-8)), 1)
                kly = torch.sum(Y * (torch.log(Y + 1e-8) - torch.log(M2 + 1e-8)), 1)
                v2 += (1 - 0.5 * (klq + kly)).sum().item()
                vc += yb.size(0)
        v_loss_avg = v_loss / len(val_ds)
        v1_avg = v1 / vc
        v2_avg = v2 / vc
        mai_v = 2 * v1_avg * v2_avg / (v1_avg + v2_avg + 1e-8)
        val_losses.append(v_loss_avg)
        val_s1.append(v1_avg)
        val_s2.append(v2_avg)
        val_mai.append(mai_v)
        print(
            f"Dataset {dataset_name} Epoch {epoch}: validation_loss = {v_loss_avg:.4f}, MAI = {mai_v:.4f}"
        )
    # record predictions
    preds, gts = [], []
    ai_model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds.append(torch.argmax(ai_model(xb), 1).cpu().numpy())
            gts.append(yb.numpy())
    experiment_data[dataset_name] = {
        "metrics": {"train": train_mai, "val": val_mai},
        "losses": {"train": train_losses, "val": val_losses},
        "sub_scores": {
            "train1": train_s1,
            "train2": train_s2,
            "val1": val_s1,
            "val2": val_s2,
        },
        "predictions": np.concatenate(preds, 0),
        "ground_truth": np.concatenate(gts, 0),
    }

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
