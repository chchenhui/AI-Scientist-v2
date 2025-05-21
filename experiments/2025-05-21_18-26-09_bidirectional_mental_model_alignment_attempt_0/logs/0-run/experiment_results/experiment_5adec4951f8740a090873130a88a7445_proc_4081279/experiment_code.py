import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.manual_seed(42)
np.random.seed(42)

# backbone
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()


# simple classifier
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# experiment
temps = [0.5, 1.0, 2.0]
datasets = ["ag_news", "yelp_polarity", "dbpedia_14"]
experiment_data = {}

for T in temps:
    key = f"softmax_temperature_{T}"
    experiment_data[key] = {}
    for name in datasets:
        # prepare data
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

        # models & optim
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
        optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # record dict
        record = {
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "accuracy": {"train": [], "val": []},
            "mai": [],
            "predictions": [],
            "ground_truth": [],
        }
        experiment_data[key][name] = record

        # training loop
        for epoch in range(1, 4):
            ai_model.train()
            user_model.train()
            tot_l, tot_a, tot_acc, n = 0.0, 0.0, 0, 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    emb = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state[:, 0, :]
                logits_ai = ai_model(emb)
                logits_user = user_model(emb)
                loss_ai = loss_fn(logits_ai, batch["label"])
                loss_user = loss_fn(logits_user, batch["label"])
                optimizer_ai.zero_grad()
                loss_ai.backward()
                optimizer_ai.step()
                optimizer_user.zero_grad()
                loss_user.backward()
                optimizer_user.step()
                bs = batch["label"].size(0)
                tot_l += loss_ai.item() * bs
                P = F.softmax(logits_ai / T, dim=1)
                Q = F.softmax(logits_user / T, dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                tot_a += torch.sum(1 - jsd).item()
                tot_acc += (torch.argmax(logits_user, 1) == batch["label"]).sum().item()
                n += bs
            # append train metrics
            record["losses"]["train"].append(tot_l / len(train_ds))
            record["alignments"]["train"].append(tot_a / n)
            record["accuracy"]["train"].append(tot_acc / n)

            # validation
            ai_model.eval()
            user_model.eval()
            v_l, v_a, v_acc, v_n = 0.0, 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    emb = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state[:, 0, :]
                    logits_ai = ai_model(emb)
                    v_l += loss_fn(logits_ai, batch["label"]).item() * batch[
                        "label"
                    ].size(0)
                    P = F.softmax(logits_ai / T, dim=1)
                    Q = F.softmax(user_model(emb) / T, dim=1)
                    M = 0.5 * (P + Q)
                    kl1 = torch.sum(
                        P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1
                    )
                    kl2 = torch.sum(
                        Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1
                    )
                    jsd = 0.5 * (kl1 + kl2)
                    v_a += torch.sum(1 - jsd).item()
                    v_acc += (
                        (torch.argmax(user_model(emb), 1) == batch["label"])
                        .sum()
                        .item()
                    )
                    v_n += batch["label"].size(0)
            val_loss = v_l / len(val_ds)
            val_align = v_a / v_n
            val_acc = v_acc / v_n
            mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)
            record["losses"]["val"].append(val_loss)
            record["alignments"]["val"].append(val_align)
            record["accuracy"]["val"].append(val_acc)
            record["mai"].append(mai)
            print(
                f"T={T} {name} Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, MAI={mai:.4f}"
            )

        # final preds & gts
        preds, gts = [], []
        ai_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                emb = distilbert(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state[:, 0, :]
                preds.append(torch.argmax(ai_model(emb), 1).cpu().numpy())
                gts.append(batch["label"].cpu().numpy())
        record["predictions"] = np.concatenate(preds)
        record["ground_truth"] = np.concatenate(gts)

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
