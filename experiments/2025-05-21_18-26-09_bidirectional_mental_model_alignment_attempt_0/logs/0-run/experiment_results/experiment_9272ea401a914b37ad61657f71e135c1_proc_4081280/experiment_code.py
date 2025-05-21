import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


# simple 2-layer MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# store results
experiment_data = {"baseline": {}, "fine_tune": {}}

for ablation in ["baseline", "fine_tune"]:
    for name in ["ag_news", "yelp_polarity", "dbpedia_14"]:
        # init storage
        experiment_data[ablation][name] = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "mai": [],
            "predictions": None,
            "ground_truth": None,
        }
        # load & tokenize
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
        train_ds.set_format("torch", ["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", ["input_ids", "attention_mask", "label"])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        num_labels = len(set(train_ds["label"]))

        # models
        distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(
            device
        )
        if ablation == "baseline":
            distilbert.eval()
            for p in distilbert.parameters():
                p.requires_grad = False
        else:
            distilbert.train()
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)

        # optimizers
        if ablation == "baseline":
            optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
        else:
            optimizer_ai = torch.optim.Adam(
                [
                    {"params": distilbert.parameters(), "lr": 2e-5},
                    {"params": ai_model.parameters(), "lr": 1e-3},
                ]
            )
        optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # training & validation
        for epoch in range(1, 4):
            # ---- train ----
            if ablation == "fine_tune":
                distilbert.train()
            ai_model.train()
            user_model.train()
            tot_loss = tot_align = tot_acc = tot_n = 0.0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # encode
                if ablation == "baseline":
                    with torch.no_grad():
                        emb = distilbert(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).last_hidden_state[:, 0, :]
                else:
                    emb = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state[:, 0, :]
                # AI head
                logits_ai = ai_model(emb)
                loss_ai = loss_fn(logits_ai, batch["label"])
                optimizer_ai.zero_grad()
                loss_ai.backward()
                optimizer_ai.step()
                # user head
                emb_user = emb.detach()
                logits_user = user_model(emb_user)
                loss_user = loss_fn(logits_user, batch["label"])
                optimizer_user.zero_grad()
                loss_user.backward()
                optimizer_user.step()
                # metrics accumulate
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
                tot_n += bs

            # record train metrics
            experiment_data[ablation][name]["losses"]["train"].append(
                tot_loss / len(train_ds)
            )
            experiment_data[ablation][name]["alignments"]["train"].append(
                tot_align / tot_n
            )
            experiment_data[ablation][name]["metrics"]["train"].append(tot_acc / tot_n)

            # ---- validation ----
            distilbert.eval()
            ai_model.eval()
            user_model.eval()
            v_loss = v_align = v_acc = v_n = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    emb = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state[:, 0, :]
                    logits_ai = ai_model(emb)
                    bs = batch["label"].size(0)
                    v_loss += loss_fn(logits_ai, batch["label"]).item() * bs
                    P = F.softmax(logits_ai, dim=1)
                    qu = user_model(emb)
                    Q = F.softmax(qu, dim=1)
                    M = 0.5 * (P + Q)
                    kl1 = torch.sum(
                        P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1
                    )
                    kl2 = torch.sum(
                        Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1
                    )
                    jsd = 0.5 * (kl1 + kl2)
                    v_align += torch.sum(1 - jsd).item()
                    v_acc += (torch.argmax(qu, dim=1) == batch["label"]).sum().item()
                    v_n += bs

            val_loss = v_loss / len(val_ds)
            val_align = v_align / v_n
            val_acc = v_acc / v_n
            experiment_data[ablation][name]["losses"]["val"].append(val_loss)
            experiment_data[ablation][name]["alignments"]["val"].append(val_align)
            experiment_data[ablation][name]["metrics"]["val"].append(val_acc)
            mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)
            experiment_data[ablation][name]["mai"].append(mai)
            print(
                f"Ablation {ablation} Dataset {name} Epoch {epoch}: validation_loss = {val_loss:.4f}, Bidirectional Alignment = {val_align:.4f}, MAI = {mai:.4f}"
            )

        # final predictions & ground truth
        preds, gts = [], []
        distilbert.eval()
        ai_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                emb = distilbert(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state[:, 0, :]
                preds.append(torch.argmax(ai_model(emb), dim=1).cpu().numpy())
                gts.append(batch["label"].cpu().numpy())
        experiment_data[ablation][name]["predictions"] = np.concatenate(preds)
        experiment_data[ablation][name]["ground_truth"] = np.concatenate(gts)

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
