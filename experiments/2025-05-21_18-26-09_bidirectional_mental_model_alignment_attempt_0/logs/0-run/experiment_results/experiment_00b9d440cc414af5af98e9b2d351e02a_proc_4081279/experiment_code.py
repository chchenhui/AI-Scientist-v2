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
torch.manual_seed(42)
np.random.seed(42)

# load model & tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
distilbert = (
    DistilBertModel.from_pretrained("distilbert-base-uncased").to(device).eval()
)

# compute head importance via L2 norm of output‐projection weights
n_layers = len(distilbert.transformer.layer)
n_heads = distilbert.transformer.layer[0].attention.n_heads
head_dim = distilbert.config.hidden_size // n_heads
head_importance = {}
for l in range(n_layers):
    W = distilbert.transformer.layer[l].attention.out_lin.weight.data.cpu()
    imp = []
    for h in range(n_heads):
        block = W[:, h * head_dim : (h + 1) * head_dim]
        imp.append(torch.norm(block).item())
    head_importance[l] = np.array(imp)

# prepare head masks
ablation_types = ["random", "importance"]
head_counts = [12, 8, 4, 2]
head_masks = {t: {} for t in ablation_types}
for t in ablation_types:
    for hc in head_counts:
        mask = torch.zeros(n_layers, n_heads)
        for l in range(n_layers):
            if t == "random":
                keep = np.random.choice(n_heads, hc, replace=False)
            else:  # importance
                keep = np.argsort(-head_importance[l])[:hc]
            mask[l, keep] = 1.0
        head_masks[t][hc] = mask.to(device)

# load datasets once
dataset_names = ["ag_news", "yelp_polarity", "dbpedia_14"]
train_loaders, val_loaders, num_labels = {}, {}, {}
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
    train_loaders[name] = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loaders[name] = DataLoader(val_ds, batch_size=32)
    num_labels[name] = len(set(train_ds["label"]))


# simple MLP
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# init data structure
experiment_data = {
    t: {
        name: {
            "losses": {"train": [], "val": []},
            "alignments": {"train": [], "val": []},
            "mai": [],
            "head_counts": [],
            "predictions": {},
            "ground_truth": None,
        }
        for name in dataset_names
    }
    for t in ablation_types
}

# run ablations
for t in ablation_types:
    for hc in head_counts:
        mask = head_masks[t][hc]
        for name in dataset_names:
            train_loader = train_loaders[name]
            val_loader = val_loaders[name]
            nl = num_labels[name]
            ai_model = MLP(distilbert.config.hidden_size, 128, nl).to(device)
            user_model = MLP(distilbert.config.hidden_size, 128, nl).to(device)
            opt_ai = torch.optim.Adam(ai_model.parameters(), lr=1e-3)
            opt_user = torch.optim.Adam(user_model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            # epochs
            for epoch in range(1, 4):
                ai_model.train()
                user_model.train()
                tot_loss, tot_align, tot_acc, n = 0.0, 0.0, 0, 0
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        out = distilbert(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            head_mask=mask,
                        )
                        emb = out.last_hidden_state[:, 0, :]
                    logits_ai = ai_model(emb)
                    logits_user = user_model(emb)
                    loss_ai = loss_fn(logits_ai, batch["label"])
                    loss_user = loss_fn(logits_user, batch["label"])
                    opt_ai.zero_grad()
                    loss_ai.backward()
                    opt_ai.step()
                    opt_user.zero_grad()
                    loss_user.backward()
                    opt_user.step()
                    bs = batch["label"].size(0)
                    tot_loss += loss_ai.item() * bs
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
                    tot_align += torch.sum(1 - jsd).item()
                    tot_acc += (
                        (torch.argmax(logits_user, dim=1) == batch["label"])
                        .sum()
                        .item()
                    )
                    n += bs
                train_loss = tot_loss / len(train_loader.dataset)
                train_align = tot_align / n
                experiment_data[t][name]["losses"]["train"].append(train_loss)
                experiment_data[t][name]["alignments"]["train"].append(train_align)

                # validation
                ai_model.eval()
                user_model.eval()
                v_loss, v_align, v_acc, v_n = 0.0, 0.0, 0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        out = distilbert(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            head_mask=mask,
                        )
                        emb = out.last_hidden_state[:, 0, :]
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
                val_loss = v_loss / len(val_loader.dataset)
                val_align = v_align / v_n
                val_acc = v_acc / v_n
                mai = 2 * (val_align * val_acc) / (val_align + val_acc + 1e-8)
                experiment_data[t][name]["losses"]["val"].append(val_loss)
                experiment_data[t][name]["alignments"]["val"].append(val_align)
                experiment_data[t][name]["mai"].append(mai)
                experiment_data[t][name]["head_counts"].append(hc)
                print(
                    f"Ablation {t} heads={hc} ds={name} epoch={epoch}: val_loss={val_loss:.4f}, MAI={mai:.4f}"
                )

            # collect predictions & ground truth
            preds, gts = [], []
            ai_model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = distilbert(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        head_mask=mask,
                    )
                    emb = out.last_hidden_state[:, 0, :]
                    preds.append(torch.argmax(ai_model(emb), dim=1).cpu().numpy())
                    gts.append(batch["label"].cpu().numpy())
            preds = np.concatenate(preds)
            gts = np.concatenate(gts)
            experiment_data[t][name]["predictions"][str(hc)] = preds
            experiment_data[t][name]["ground_truth"] = gts

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
