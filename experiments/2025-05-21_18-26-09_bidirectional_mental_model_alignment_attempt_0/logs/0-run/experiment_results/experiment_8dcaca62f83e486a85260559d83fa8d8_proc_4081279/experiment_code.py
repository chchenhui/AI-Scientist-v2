import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# Setup working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Seeds
torch.manual_seed(42)
np.random.seed(42)

# Tokenizer and base model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()


# Simple 2-layer MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# Ablation hooks
ablations = {
    "layer_1": lambda hs: hs[1][:, 0, :],
    "layer_3": lambda hs: hs[3][:, 0, :],
    "layer_5": lambda hs: hs[5][:, 0, :],
    "avg_last2": lambda hs: (hs[-1][:, 0, :] + hs[-2][:, 0, :]) / 2.0,
}

# Datasets
dataset_names = ["ag_news", "yelp_polarity", "dbpedia_14"]
data_loaders = {}
for name in dataset_names:
    raw = load_dataset(name, split="train").shuffle(seed=0).select(range(2500))
    split = raw.train_test_split(test_size=0.2, seed=0)
    train_ds, val_ds = split["train"], split["test"]
    text_key = "text" if "text" in raw.column_names else "content"

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_key], padding="max_length", truncation=True, max_length=128
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_key])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=[text_key])
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    num_labels = len(set(train_ds["label"]))
    data_loaders[name] = (train_ds, val_ds, train_loader, val_loader, num_labels)

# Loss
loss_fn = nn.CrossEntropyLoss()
ln2 = math.log(2.0)

# Main experiment
experiment_data = {}
for ablation_name, ablate in ablations.items():
    experiment_data[ablation_name] = {}
    for dname in dataset_names:
        train_ds, val_ds, train_loader, val_loader, num_labels = data_loaders[dname]
        # Prepare result containers
        results = {
            "losses": {"train": [], "val": []},
            "metrics": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        # Instantiate models & optimizers
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        opt_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
        opt_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)

        # Train epochs
        for epoch in range(1, 4):
            ai_model.train()
            user_model.train()
            tot_loss = 0.0
            tot_align1 = 0.0
            tot_align2 = 0.0
            n = 0
            for batch in train_loader:
                # move to device
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                with torch.no_grad():
                    out = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )
                    hs = out.hidden_states
                    emb = ablate(hs)
                # Forward
                logits_ai = ai_model(emb)
                logits_user = user_model(emb)
                # Loss and backward
                loss_ai = loss_fn(logits_ai, batch["label"])
                loss_user = loss_fn(logits_user, batch["label"])
                opt_ai.zero_grad()
                loss_ai.backward()
                opt_ai.step()
                opt_user.zero_grad()
                loss_user.backward()
                opt_user.step()
                # Metrics
                bs = batch["label"].size(0)
                tot_loss += loss_ai.item() * bs
                P = F.softmax(logits_ai, dim=1)
                Q = F.softmax(logits_user, dim=1)
                # measure1: AI vs ground-truth one-hot
                onehot = F.one_hot(batch["label"], num_labels).float()
                M1 = 0.5 * (P + onehot)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M1 + 1e-8)), dim=1)
                kl2 = torch.sum(
                    onehot * (torch.log(onehot + 1e-8) - torch.log(M1 + 1e-8)), dim=1
                )
                jsd1 = 0.5 * (kl1 + kl2)
                sim1 = 1 - (jsd1 / ln2)
                # measure2: user vs AI
                M2 = 0.5 * (Q + P)
                kl1b = torch.sum(
                    Q * (torch.log(Q + 1e-8) - torch.log(M2 + 1e-8)), dim=1
                )
                kl2b = torch.sum(
                    P * (torch.log(P + 1e-8) - torch.log(M2 + 1e-8)), dim=1
                )
                jsd2 = 0.5 * (kl1b + kl2b)
                sim2 = 1 - (jsd2 / ln2)
                tot_align1 += sim1.sum().item()
                tot_align2 += sim2.sum().item()
                n += bs
            # Record train stats
            avg_loss = tot_loss / len(train_ds)
            avg_sim1 = tot_align1 / n
            avg_sim2 = tot_align2 / n
            bmsa_train = 0.5 * (avg_sim1 + avg_sim2)
            results["losses"]["train"].append(avg_loss)
            results["metrics"]["train"].append(bmsa_train)

            # Validation
            ai_model.eval()
            user_model.eval()
            v_loss = 0.0
            v_a1 = 0.0
            v_a2 = 0.0
            v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {
                        k: v.to(device)
                        for k, v in batch.items()
                        if isinstance(v, torch.Tensor)
                    }
                    out = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                    )
                    hs = out.hidden_states
                    emb = ablate(hs)
                    logits_ai = ai_model(emb)
                    v_loss += loss_fn(logits_ai, batch["label"]).item() * batch[
                        "label"
                    ].size(0)
                    P = F.softmax(logits_ai, dim=1)
                    Q = F.softmax(user_model(emb), dim=1)
                    onehot = F.one_hot(batch["label"], num_labels).float()
                    M1 = 0.5 * (P + onehot)
                    kl1 = torch.sum(
                        P * (torch.log(P + 1e-8) - torch.log(M1 + 1e-8)), dim=1
                    )
                    kl2 = torch.sum(
                        onehot * (torch.log(onehot + 1e-8) - torch.log(M1 + 1e-8)),
                        dim=1,
                    )
                    sim1 = 1 - (0.5 * (kl1 + kl2) / ln2)
                    M2 = 0.5 * (Q + P)
                    kl1b = torch.sum(
                        Q * (torch.log(Q + 1e-8) - torch.log(M2 + 1e-8)), dim=1
                    )
                    kl2b = torch.sum(
                        P * (torch.log(P + 1e-8) - torch.log(M2 + 1e-8)), dim=1
                    )
                    sim2 = 1 - (0.5 * (kl1b + kl2b) / ln2)
                    v_a1 += sim1.sum().item()
                    v_a2 += sim2.sum().item()
                    v_n += batch["label"].size(0)
            val_loss = v_loss / len(val_ds)
            avg_sim1_v = v_a1 / v_n
            avg_sim2_v = v_a2 / v_n
            bmsa_val = 0.5 * (avg_sim1_v + avg_sim2_v)
            results["losses"]["val"].append(val_loss)
            results["metrics"]["val"].append(bmsa_val)
            print(
                f"{ablation_name} | {dname} | Epoch {epoch}: validation_loss = {val_loss:.4f}, Bidirectional Alignment = {bmsa_val:.4f}"
            )

        # Final predictions
        ai_model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                out = distilbert(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
                emb = ablate(out.hidden_states)
                preds.append(torch.argmax(ai_model(emb), dim=1).cpu().numpy())
                gts.append(batch["label"].cpu().numpy())
        results["predictions"] = np.concatenate(preds)
        results["ground_truth"] = np.concatenate(gts)
        experiment_data[ablation_name][dname] = results

# Save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
