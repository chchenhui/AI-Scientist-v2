import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# Setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# Frozen encoder
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()


# Simple MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# Ablation: optimizer choices
optimizers = {
    "adam": lambda params: torch.optim.Adam(params, lr=1e-3),
    "sgd": lambda params: torch.optim.SGD(params, lr=1e-3, momentum=0.9),
    "adamw": lambda params: torch.optim.AdamW(params, lr=1e-3),
}

datasets_list = ["ag_news", "yelp_polarity", "dbpedia_14"]
loss_fn = nn.CrossEntropyLoss()
experiment_data = {}

for opt_name, opt_fn in optimizers.items():
    experiment_data[opt_name] = {}
    for name in datasets_list:
        # Load and preprocess
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

        # Storage
        experiment_data[opt_name][name] = {
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "accuracy": {"train": [], "val": []},
            "mai": [],
            "predictions": [],
            "ground_truth": [],
        }

        # Models and optimizers
        num_labels = len(set(train_ds["label"]))
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        optimizer_ai = opt_fn(ai_model.parameters())
        optimizer_user = opt_fn(user_model.parameters())

        # Train epochs
        for epoch in range(1, 4):
            ai_model.train()
            user_model.train()
            tot_loss = tot_align = tot_acc = n = 0.0
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
            train_align = tot_align / n
            train_acc = tot_acc / n
            experiment_data[opt_name][name]["losses"]["train"].append(train_loss)
            experiment_data[opt_name][name]["alignments"]["train"].append(train_align)
            experiment_data[opt_name][name]["accuracy"]["train"].append(train_acc)

            # Validation
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
                    logits_user = user_model(emb)
                    bs = batch["label"].size(0)
                    v_loss += loss_fn(logits_ai, batch["label"]).item() * bs
                    P = F.softmax(logits_ai, dim=1)
                    Q = F.softmax(logits_user, dim=1)
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
                        (torch.argmax(logits_user, dim=1) == batch["label"])
                        .sum()
                        .item()
                    )
                    v_n += bs
            val_loss = v_loss / len(val_ds)
            val_align = v_align / v_n
            val_acc = v_acc / v_n
            mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)
            experiment_data[opt_name][name]["losses"]["val"].append(val_loss)
            experiment_data[opt_name][name]["alignments"]["val"].append(val_align)
            experiment_data[opt_name][name]["accuracy"]["val"].append(val_acc)
            experiment_data[opt_name][name]["mai"].append(mai)
            print(
                f"Opt {opt_name} | {name} | Epoch {epoch} | val_loss={val_loss:.4f} val_align={val_align:.4f} val_acc={val_acc:.4f} MAI={mai:.4f}"
            )

        # Collect predictions
        preds, gts = [], []
        ai_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                emb = distilbert(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state[:, 0, :]
                preds.append(torch.argmax(ai_model(emb), dim=1).cpu().numpy())
                gts.append(batch["label"].cpu().numpy())
        experiment_data[opt_name][name]["predictions"] = np.concatenate(preds)
        experiment_data[opt_name][name]["ground_truth"] = np.concatenate(gts)

# Save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
