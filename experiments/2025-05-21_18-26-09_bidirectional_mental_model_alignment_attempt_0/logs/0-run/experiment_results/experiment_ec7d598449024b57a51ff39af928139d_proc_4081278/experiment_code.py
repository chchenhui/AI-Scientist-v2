import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# setup working directory and seeds
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# load tokenizer and frozen DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()


# simple MLP head
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# token dropout helper (masks tokens with [MASK])
def apply_token_dropout(input_ids, attention_mask, drop_rate):
    if drop_rate <= 0:
        return input_ids, attention_mask
    mask = torch.rand(input_ids.shape, device=input_ids.device) < drop_rate
    mask &= attention_mask.bool()
    mask[:, 0] = False  # never mask [CLS]
    dropped = input_ids.clone()
    dropped[mask] = tokenizer.mask_token_id
    return dropped, attention_mask


# ablation rates and datasets
ablation_rates = [0.0, 0.1, 0.2, 0.3]
datasets = ["ag_news", "yelp_polarity", "dbpedia_14"]
experiment_data = {}

for drop in ablation_rates:
    key = f"token_dropout_{int(drop*100)}"
    experiment_data[key] = {}
    for name in datasets:
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

        # initialize models and optimizers
        num_labels = len(set(train_ds["label"]))
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
        optimizer_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        # storage for this dataset & ablation
        data = {
            "metrics": {"train": [], "val": []},  # classification accuracies
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "mai": [],  # val MAI per epoch
            "predictions": [],
            "ground_truth": [],
        }

        # training loop
        for epoch in range(1, 4):
            ai_model.train()
            user_model.train()
            tot_loss = tot_align = tot_acc = n_samples = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                ids, att = apply_token_dropout(
                    batch["input_ids"], batch["attention_mask"], drop
                )
                with torch.no_grad():
                    emb = distilbert(
                        input_ids=ids, attention_mask=att
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
                n_samples += bs

            loss_train = tot_loss / len(train_ds)
            align_train = tot_align / n_samples
            acc_train = tot_acc / n_samples
            data["losses"]["train"].append(loss_train)
            data["alignments"]["train"].append(align_train)
            data["metrics"]["train"].append(acc_train)

            # validation
            ai_model.eval()
            user_model.eval()
            v_loss = v_align = v_acc = v_n = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ids, att = apply_token_dropout(
                        batch["input_ids"], batch["attention_mask"], drop
                    )
                    emb = distilbert(
                        input_ids=ids, attention_mask=att
                    ).last_hidden_state[:, 0, :]
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

            loss_val = v_loss / len(val_ds)
            align_val = v_align / v_n
            acc_val = v_acc / v_n
            mai_val = 2 * (align_val * acc_val) / (align_val + acc_val + 1e-8)
            data["losses"]["val"].append(loss_val)
            data["alignments"]["val"].append(align_val)
            data["metrics"]["val"].append(acc_val)
            data["mai"].append(mai_val)

            print(
                f"{key} {name} Epoch {epoch}: val_loss={loss_val:.4f}, "
                f"val_acc={acc_val:.4f}, val_align={align_val:.4f}, MAI={mai_val:.4f}"
            )

        # final predictions & ground truth
        preds, gts = [], []
        ai_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                ids, att = apply_token_dropout(
                    batch["input_ids"], batch["attention_mask"], drop
                )
                emb = distilbert(input_ids=ids, attention_mask=att).last_hidden_state[
                    :, 0, :
                ]
                preds.append(torch.argmax(ai_model(emb), dim=1).cpu().numpy())
                gts.append(batch["label"].cpu().numpy())
        data["predictions"] = np.concatenate(preds)
        data["ground_truth"] = np.concatenate(gts)
        experiment_data[key][name] = data

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
