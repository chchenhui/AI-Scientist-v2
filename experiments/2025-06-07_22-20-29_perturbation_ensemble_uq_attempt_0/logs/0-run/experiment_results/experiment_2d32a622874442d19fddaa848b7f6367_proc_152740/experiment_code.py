import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random, re
import nltk
import torch
import numpy as np
import torch.nn.functional as F
from nltk.corpus import wordnet
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# prepare WordNet
nltk.download("wordnet", quiet=True)


def generate_paraphrases(text, K):
    words = text.split()
    paras = []
    for _ in range(K):
        new = words.copy()
        for idx in random.sample(range(len(words)), min(2, len(words))):
            w = re.sub(r"\W+", "", words[idx])
            syns = wordnet.synsets(w)
            lemmas = {
                l.name().replace("_", " ")
                for s in syns
                for l in s.lemmas()
                if l.name().lower() != w.lower()
            }
            if lemmas:
                new[idx] = random.choice(list(lemmas))
        paras.append(" ".join(new))
    return paras


# datasets
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}

# hyperparams
train_size, val_size = 5000, 500
K, epochs, bs, lr = 5, 5, 32, 2e-5

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
experiment_data = {}

for name, (ds, sub, text_col, label_col) in datasets_info.items():
    # load and trim
    ds_train = (
        load_dataset(ds, sub, split="train") if sub else load_dataset(ds, split="train")
    )
    ds_val = load_dataset(ds, sub, split="validation" if sub else "test")
    ds_train = ds_train.shuffle(42).select(range(train_size))
    ds_val = ds_val.shuffle(42).select(range(val_size))
    texts_train, labels_train = ds_train[text_col], ds_train[label_col]
    texts_val, labels_val = ds_val[text_col], ds_val[label_col]
    # paraphrases for val
    paras = {i: generate_paraphrases(t, K) for i, t in enumerate(texts_val)}
    # tokenize
    tr_enc = tokenizer(texts_train, truncation=True, padding=True, return_tensors="pt")
    va_enc = tokenizer(texts_val, truncation=True, padding=True, return_tensors="pt")
    train_ds = TensorDataset(
        tr_enc["input_ids"], tr_enc["attention_mask"], torch.tensor(labels_train)
    )
    val_ds = TensorDataset(
        va_enc["input_ids"], va_enc["attention_mask"], torch.tensor(labels_val)
    )
    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=bs)

    # model & optimizer
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    experiment_data[name] = {
        "losses": {"train": [], "val": []},
        "metrics": {"detection": []},
        "predictions": [],
        "ground_truth": labels_val,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for ids, mask, labels in tr_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            train_losses.append(out.loss.item())
        experiment_data[name]["losses"]["train"].append(
            {"epoch": epoch, "loss": float(np.mean(train_losses))}
        )

        # validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for ids, mask, labels in va_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                val_losses.append(out.loss.item())
        val_loss = float(np.mean(val_losses))
        experiment_data[name]["losses"]["val"].append(
            {"epoch": epoch, "loss": val_loss}
        )
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

        # detection metrics
        uncs_vote, uncs_kl, errs = [], [], []
        for i, txt in enumerate(texts_val):
            probs = []
            preds = []
            for variant in [txt] + paras[i]:
                enc = tokenizer(
                    variant, return_tensors="pt", truncation=True, padding=True
                ).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                p = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                probs.append(p)
                preds.append(int(p.argmax().item()))
            maj = max(set(preds), key=preds.count)
            uncs_vote.append(1 - preds.count(maj) / len(preds))
            # sym KL across all pairs
            kl_vals = []
            for a in range(len(probs)):
                for b in range(a + 1, len(probs)):
                    P, Q = probs[a], probs[b]
                    kl1 = F.kl_div(Q.log(), P, reduction="sum").item()
                    kl2 = F.kl_div(P.log(), Q, reduction="sum").item()
                    kl_vals.append(0.5 * (kl1 + kl2))
            uncs_kl.append(float(np.mean(kl_vals)))
            errs.append(int(preds[0] != int(labels_val[i])))
        # compute AUCs and DES
        try:
            auc_v = roc_auc_score(errs, uncs_vote)
        except:
            auc_v = 0.5
        try:
            auc_k = roc_auc_score(errs, uncs_kl)
        except:
            auc_k = 0.5
        des_v = auc_v / (K + 1)
        des_k = auc_k / (K + 1)
        experiment_data[name]["metrics"]["detection"].append(
            {
                "epoch": epoch,
                "auc_vote": auc_v,
                "DES_vote": des_v,
                "auc_kl": auc_k,
                "DES_kl": des_k,
            }
        )
        print(
            f"Epoch {epoch}: AUC_vote={auc_v:.4f}, DES_vote={des_v:.4f}, AUC_kl={auc_k:.4f}, DES_kl={des_k:.4f}"
        )

    # final predictions
    preds = []
    model.eval()
    with torch.no_grad():
        for ids, mask, _ in va_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
    experiment_data[name]["predictions"] = preds

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
