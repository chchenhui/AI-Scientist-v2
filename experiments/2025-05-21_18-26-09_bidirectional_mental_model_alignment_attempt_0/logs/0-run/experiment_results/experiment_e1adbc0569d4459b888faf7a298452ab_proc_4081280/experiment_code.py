import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

# set up
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# global seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# tokenizer + frozen encoder
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
distilbert.eval()


# simple two‐layer MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# hyperparameters
datasets = ["ag_news", "yelp_polarity", "dbpedia_14"]
lambdas = [0.0, 0.1, 0.5, 1.0]
num_epochs = 3
batch_size = 32
lr = 1e-3
loss_fn = nn.CrossEntropyLoss()

# container for everything
experiment_data = {}

for lam in lambdas:
    # reseed so each lambda run starts from identical MLP init
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    lam_key = f"lambda_{str(lam).replace('.', '_')}"
    experiment_data[lam_key] = {}
    print(f"\n=== Starting ablation {lam_key} ===")

    for name in datasets:
        print(f"\nDataset {name}, λ={lam}")
        # prepare storage
        experiment_data[lam_key][name] = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "mai": [],
            "predictions": [],
            "ground_truth": [],
        }

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
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # models + optimizers
        num_labels = len(set(train_ds["label"]))
        ai_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        user_model = MLP(distilbert.config.hidden_size, 128, num_labels).to(device)
        optimizer_ai = torch.optim.Adam(ai_model.parameters(), lr=lr)
        optimizer_user = torch.optim.Adam(user_model.parameters(), lr=lr)

        # training epochs
        for epoch in range(1, num_epochs + 1):
            ai_model.train()
            user_model.train()
            tot_loss, tot_align, tot_acc, n = 0.0, 0.0, 0, 0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # embed
                with torch.no_grad():
                    emb = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).last_hidden_state[:, 0, :]
                # forward
                logits_ai = ai_model(emb)
                logits_user = user_model(emb)
                # CE losses
                loss_ai = loss_fn(logits_ai, batch["label"])
                loss_user = loss_fn(logits_user, batch["label"])
                # JSD penalty
                P = F.softmax(logits_ai, dim=1)
                Q = F.softmax(logits_user, dim=1)
                M = 0.5 * (P + Q)
                kl1 = torch.sum(P * (torch.log(P + 1e-8) - torch.log(M + 1e-8)), dim=1)
                kl2 = torch.sum(Q * (torch.log(Q + 1e-8) - torch.log(M + 1e-8)), dim=1)
                jsd = 0.5 * (kl1 + kl2)
                loss_reg = jsd.mean()

                # joint backward + step
                optimizer_ai.zero_grad()
                optimizer_user.zero_grad()
                total_loss = loss_ai + loss_user + lam * loss_reg
                total_loss.backward()
                optimizer_ai.step()
                optimizer_user.step()

                bs = batch["label"].size(0)
                tot_loss += loss_ai.item() * bs
                tot_align += torch.sum(1 - jsd).item()
                tot_acc += (
                    (torch.argmax(logits_user, dim=1) == batch["label"]).sum().item()
                )
                n += bs

            # record train stats
            experiment_data[lam_key][name]["losses"]["train"].append(
                tot_loss / len(train_ds)
            )
            experiment_data[lam_key][name]["alignments"]["train"].append(tot_align / n)
            experiment_data[lam_key][name]["metrics"]["train"].append(tot_acc / n)

            # validation
            ai_model.eval()
            user_model.eval()
            v_loss, v_align, v_acc, v_n = 0.0, 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    emb = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
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
            val_loss = v_loss / len(val_ds)
            val_align = v_align / v_n
            val_acc = v_acc / v_n
            mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)

            # record val stats
            experiment_data[lam_key][name]["losses"]["val"].append(val_loss)
            experiment_data[lam_key][name]["alignments"]["val"].append(val_align)
            experiment_data[lam_key][name]["metrics"]["val"].append(val_acc)
            experiment_data[lam_key][name]["mai"].append(mai)

            print(
                f"  Ep{epoch}: train_loss={tot_loss/len(train_ds):.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_align={val_align:.4f}, MAI={mai:.4f}"
            )

        # final predictions by ai_model
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
        experiment_data[lam_key][name]["predictions"] = np.concatenate(preds)
        experiment_data[lam_key][name]["ground_truth"] = np.concatenate(gts)

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("\nSaved experiment_data.npy")
