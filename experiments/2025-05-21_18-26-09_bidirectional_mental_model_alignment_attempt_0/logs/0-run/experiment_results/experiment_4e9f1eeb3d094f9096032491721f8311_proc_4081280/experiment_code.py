import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# model classes
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class MLPShared1(nn.Module):
    def __init__(self, shared_fc1, fc2):
        super().__init__()
        self.fc1 = shared_fc1
        self.fc2 = fc2

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# settings
datasets = ["ag_news", "yelp_polarity", "dbpedia_14"]
ablations = ["independent", "shared_fc1", "shared_fc1_fc2"]
experiment_data = {a: {} for a in ablations}
loss_fn = nn.CrossEntropyLoss()

for ablation in ablations:
    for name in datasets:
        # prepare data
        raw = load_dataset(name, split="train").shuffle(seed=0).select(range(2500))
        split = raw.train_test_split(test_size=0.2, seed=0)
        train_ds, val_ds = split["train"], split["test"]
        text_key = "text" if "text" in raw.column_names else "content"

        def tok(b):
            return tokenizer(
                b[text_key], padding="max_length", truncation=True, max_length=128
            )

        train_ds = train_ds.map(tok, batched=True, remove_columns=[text_key])
        val_ds = val_ds.map(tok, batched=True, remove_columns=[text_key])
        train_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        val_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        num_labels = len(set(train_ds["label"]))
        # build models & optimizers
        in_dim, hid_dim, out_dim = distilbert.config.hidden_size, 128, num_labels
        if ablation == "independent":
            ai_model = MLP(in_dim, hid_dim, out_dim).to(device)
            user_model = MLP(in_dim, hid_dim, out_dim).to(device)
            opt_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
            opt_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
            optimizer = None
        elif ablation == "shared_fc1":
            shared_fc1 = nn.Linear(in_dim, hid_dim).to(device)
            ai_fc2 = nn.Linear(hid_dim, out_dim).to(device)
            user_fc2 = nn.Linear(hid_dim, out_dim).to(device)
            ai_model = MLPShared1(shared_fc1, ai_fc2).to(device)
            user_model = MLPShared1(shared_fc1, user_fc2).to(device)
            optimizer = torch.optim.Adam(
                list(shared_fc1.parameters())
                + list(ai_fc2.parameters())
                + list(user_fc2.parameters()),
                lr=1e-3,
            )
        else:  # fully shared
            shared = MLP(in_dim, hid_dim, out_dim).to(device)
            ai_model, user_model = shared, shared
            optimizer = torch.optim.Adam(shared.parameters(), lr=1e-3)
        # init logs
        ed = {
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "mai": [],
            "predictions": [],
            "ground_truth": [],
        }
        # train & validate
        for epoch in range(1, 4):
            ai_model.train()
            user_model.train()
            tot_loss = tot_align = tot_acc = n = 0
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
                # update
                if ablation == "independent":
                    opt_ai.zero_grad()
                    loss_ai.backward()
                    opt_ai.step()
                    opt_user.zero_grad()
                    loss_user.backward()
                    opt_user.step()
                else:
                    optimizer.zero_grad()
                    (loss_ai + loss_user).backward()
                    optimizer.step()
                # metrics
                bs = batch["label"].size(0)
                tot_loss += loss_ai.item() * bs
                P = F.softmax(logits_ai, dim=1)
                Q = F.softmax(logits_user, dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                tot_align += torch.sum(1 - jsd).item()
                tot_acc += (torch.argmax(logits_user, 1) == batch["label"]).sum().item()
                n += bs
            ed["losses"]["train"].append(tot_loss / len(train_ds))
            ed["alignments"]["train"].append(tot_align / n)
            # validation
            ai_model.eval()
            user_model.eval()
            v_loss = v_align = v_acc = v_n = 0
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
                        (torch.argmax(user_model(emb), 1) == batch["label"])
                        .sum()
                        .item()
                    )
                    v_n += bs
            val_loss = v_loss / len(val_ds)
            val_align = v_align / v_n
            val_acc = v_acc / v_n
            mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)
            ed["losses"]["val"].append(val_loss)
            ed["alignments"]["val"].append(val_align)
            ed["mai"].append(mai)
            print(
                f"Ablation {ablation} Dataset {name} Epoch {epoch}: "
                f"validation_loss={val_loss:.4f}, MAI={mai:.4f}"
            )
        # final preds
        ai_model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                emb = distilbert(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).last_hidden_state[:, 0, :]
                preds.append(torch.argmax(ai_model(emb), 1).cpu().numpy())
                gts.append(batch["label"].cpu().numpy())
        ed["predictions"] = np.concatenate(preds)
        ed["ground_truth"] = np.concatenate(gts)
        experiment_data[ablation][name] = ed

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
