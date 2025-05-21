import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")

# Tokenizer & frozen encoder
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()


# Simple MLP head
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction="mean", eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1).clamp(min=self.eps)
        t_one = F.one_hot(targets, inputs.size(1)).float().to(inputs.device)
        p_t = (probs * t_one).sum(1)
        loss = -((1 - p_t) ** self.gamma) * torch.log(p_t)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# Ablation settings & data structures
gammas = [1, 2, 5]
datasets = ["ag_news", "yelp_polarity", "dbpedia_14"]
experiment_data = {}

for gamma in gammas:
    ab_key = f"focal_loss_gamma_{gamma}"
    experiment_data[ab_key] = {}
    loss_fn = FocalLoss(gamma=gamma)

    for name in datasets:
        # Load & tokenize
        raw = load_dataset(name, split="train").shuffle(seed=0).select(range(2500))
        split = raw.train_test_split(test_size=0.2, seed=0)
        train_ds, val_ds = split["train"], split["test"]
        text_key = "text" if "text" in raw.column_names else "content"

        def tok(batch):
            return tokenizer(
                batch[text_key], padding="max_length", truncation=True, max_length=128
            )

        train_ds = train_ds.map(tok, batched=True, remove_columns=[text_key])
        val_ds = val_ds.map(tok, batched=True, remove_columns=[text_key])
        cols = ["input_ids", "attention_mask", "label"]
        train_ds.set_format(type="torch", columns=cols)
        val_ds.set_format(type="torch", columns=cols)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        num_labels = len(set(train_ds["label"]))

        # Models & optimizers
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        opt_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
        opt_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)

        # Storage per dataset
        experiment_data[ab_key][name] = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        # Training loop
        for epoch in range(1, 4):
            ai_model.train()
            user_model.train()
            tot_loss = tot_align = tot_acc = n = 0
            for b in train_loader:
                b = {k: v.to(device) for k, v in b.items()}
                with torch.no_grad():
                    emb = distilbert(
                        input_ids=b["input_ids"], attention_mask=b["attention_mask"]
                    ).last_hidden_state[:, 0, :]
                logits_ai = ai_model(emb)
                logits_user = user_model(emb)
                loss_ai = loss_fn(logits_ai, b["label"])
                # AI update
                opt_ai.zero_grad()
                loss_ai.backward()
                opt_ai.step()
                # User update
                loss_user = loss_fn(logits_user, b["label"])
                opt_user.zero_grad()
                loss_user.backward()
                opt_user.step()

                bs = b["label"].size(0)
                tot_loss += loss_ai.item() * bs
                P = F.softmax(logits_ai, dim=1)
                Q = F.softmax(logits_user, dim=1)
                M = 0.5 * (P + Q)
                kl1 = (P * (torch.log(P + 1e-8) - torch.log(M + 1e-8))).sum(1)
                kl2 = (Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8))).sum(1)
                jsd = 0.5 * (kl1 + kl2)
                tot_align += (1 - jsd).sum().item()
                tot_acc += (torch.argmax(logits_user, 1) == b["label"]).sum().item()
                n += bs

            # Compute train stats
            train_loss = tot_loss / len(train_ds)
            train_align = tot_align / n
            train_acc = tot_acc / n
            train_mai = 2 * (train_align * train_acc) / (train_align + train_acc + 1e-8)
            experiment_data[ab_key][name]["losses"]["train"].append(train_loss)
            experiment_data[ab_key][name]["alignments"]["train"].append(train_align)
            experiment_data[ab_key][name]["metrics"]["train"].append(train_mai)

            # Validation
            ai_model.eval()
            user_model.eval()
            v_loss = v_align = v_acc = v_n = 0
            with torch.no_grad():
                for b in val_loader:
                    b = {k: v.to(device) for k, v in b.items()}
                    emb = distilbert(
                        input_ids=b["input_ids"], attention_mask=b["attention_mask"]
                    ).last_hidden_state[:, 0, :]
                    logits_ai = ai_model(emb)
                    v_loss += loss_fn(logits_ai, b["label"]).item() * b["label"].size(0)
                    P = F.softmax(logits_ai, 1)
                    Q = F.softmax(user_model(emb), 1)
                    M = 0.5 * (P + Q)
                    kl1 = (P * (torch.log(P + 1e-8) - torch.log(M + 1e-8))).sum(1)
                    kl2 = (Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8))).sum(1)
                    jsd = 0.5 * (kl1 + kl2)
                    v_align += (1 - jsd).sum().item()
                    v_acc += (
                        (torch.argmax(user_model(emb), 1) == b["label"]).sum().item()
                    )
                    v_n += b["label"].size(0)

            val_loss = v_loss / len(val_ds)
            val_align = v_align / v_n
            val_acc = v_acc / v_n
            val_mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)
            experiment_data[ab_key][name]["losses"]["val"].append(val_loss)
            experiment_data[ab_key][name]["alignments"]["val"].append(val_align)
            experiment_data[ab_key][name]["metrics"]["val"].append(val_mai)

            print(
                f"Ablation {ab_key} Dataset {name} Epoch {epoch}: val_loss={val_loss:.4f}, MAI={val_mai:.4f}"
            )

        # Final predictions on val
        preds, gts = [], []
        ai_model.eval()
        with torch.no_grad():
            for b in val_loader:
                b = {k: v.to(device) for k, v in b.items()}
                emb = distilbert(
                    input_ids=b["input_ids"], attention_mask=b["attention_mask"]
                ).last_hidden_state[:, 0, :]
                preds.append(torch.argmax(ai_model(emb), 1).cpu().numpy())
                gts.append(b["label"].cpu().numpy())
        experiment_data[ab_key][name]["predictions"] = np.concatenate(preds)
        experiment_data[ab_key][name]["ground_truth"] = np.concatenate(gts)

# Save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
