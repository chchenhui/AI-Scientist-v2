import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# pooling functions for ablation
poolings = {
    "cls_pooling": lambda hs: hs[:, 0, :],
    "mean_pooling": lambda hs: hs.mean(dim=1),
}

experiment_data = {p: {} for p in poolings}

for pooling_name, pool_fn in poolings.items():
    for name in ["ag_news", "yelp_polarity", "dbpedia_14"]:
        # load and tokenize
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
        train_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        val_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)

        num_labels = len(set(train_ds["label"]))
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        opt_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
        opt_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # storage for this dataset+pooling
        data = {
            "metrics": {"train": [], "val": []},  # accuracy
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "mai": [],
            "predictions": [],
            "ground_truth": [],
        }

        # training
        for epoch in range(1, 4):
            ai_model.train()
            user_model.train()
            tot_loss = tot_acc = tot_align = n = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    hs = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state
                    emb = pool_fn(hs)
                logits_ai = ai_model(emb)
                logits_user = user_model(emb)
                loss_ai = loss_fn(logits_ai, batch["label"])
                loss_user = loss_fn(logits_user, batch["label"])
                # update AI
                opt_ai.zero_grad()
                loss_ai.backward()
                opt_ai.step()
                # update user
                opt_user.zero_grad()
                loss_user.backward()
                opt_user.step()
                bs = batch["label"].size(0)
                tot_loss += loss_ai.item() * bs
                P = F.softmax(logits_ai, dim=1)
                Q = F.softmax(logits_user, dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                tot_align += torch.sum(1 - jsd).item()
                tot_acc += (
                    (torch.argmax(logits_user, dim=1) == batch["label"]).sum().item()
                )
                n += bs
            train_loss = tot_loss / len(train_ds)
            train_acc = tot_acc / n
            train_align = tot_align / n
            data["losses"]["train"].append(train_loss)
            data["metrics"]["train"].append(train_acc)
            data["alignments"]["train"].append(train_align)

            # validation
            ai_model.eval()
            user_model.eval()
            v_loss = v_acc = v_align = v_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    hs = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state
                    emb = pool_fn(hs)
                    logits_ai = ai_model(emb)
                    v_loss += loss_fn(logits_ai, batch["label"]).item() * batch[
                        "label"
                    ].size(0)
                    P = F.softmax(logits_ai, dim=1)
                    Q = F.softmax(user_model(emb), dim=1)
                    M = 0.5 * (P + Q)
                    kl1 = torch.sum(
                        P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1
                    )
                    kl2 = torch.sum(
                        Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1
                    )
                    jsd = 0.5 * (kl1 + kl2)
                    v_align += torch.sum(1 - jsd).item()
                    v_acc += (
                        (torch.argmax(user_model(emb), dim=1) == batch["label"])
                        .sum()
                        .item()
                    )
                    v_n += batch["label"].size(0)
            val_loss = v_loss / len(val_ds)
            val_acc = v_acc / v_n
            val_align = v_align / v_n
            mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)
            data["losses"]["val"].append(val_loss)
            data["metrics"]["val"].append(val_acc)
            data["alignments"]["val"].append(val_align)
            data["mai"].append(mai)
            print(
                f"{pooling_name} | {name} | Epoch {epoch} -> val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_align: {val_align:.4f}, MAI: {mai:.4f}"
            )

        # final predictions & gts
        ai_model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                hs = distilbert(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state
                emb = pool_fn(hs)
                preds.append(torch.argmax(ai_model(emb), dim=1).cpu().numpy())
                gts.append(batch["label"].cpu().numpy())
        data["predictions"] = np.concatenate(preds)
        data["ground_truth"] = np.concatenate(gts)

        experiment_data[pooling_name][name] = data

# save results
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
