import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random
import re
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
from scipy.stats import spearmanr

nltk.download("wordnet", quiet=True)

# GPU/CPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


# data and hyperparameters
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}
train_size, val_size = 5000, 500
K, K_train, epochs, bs, lr = 5, 1, 5, 32, 2e-5

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
experiment_data = {}

for ablation in ["baseline", "paraphrase_aug"]:
    for name, (ds, sub, txt_col, lbl_col) in datasets_info.items():
        # load & subsample
        train_ds = (
            load_dataset(ds, sub, split="train")
            if sub
            else load_dataset(ds, split="train")
        )
        val_ds = load_dataset(ds, sub, split="validation" if sub else "test")
        train_ds = train_ds.shuffle(42).select(range(train_size))
        val_ds = val_ds.shuffle(42).select(range(val_size))
        texts_train, labels_train = train_ds[txt_col], train_ds[lbl_col]
        texts_val, labels_val = val_ds[txt_col], val_ds[lbl_col]
        paras_val = {i: generate_paraphrases(t, K) for i, t in enumerate(texts_val)}

        # prepare training data with limited augmentation
        if ablation == "baseline":
            texts_tr, labels_tr = texts_train, labels_train
        else:
            texts_tr, labels_tr = [], []
            for t, l in zip(texts_train, labels_train):
                texts_tr.append(t)
                labels_tr.append(l)
                parap = generate_paraphrases(t, K_train)[0]
                texts_tr.append(parap)
                labels_tr.append(l)

        # tokenize
        tr_enc = tokenizer(texts_tr, truncation=True, padding=True, return_tensors="pt")
        va_enc = tokenizer(
            texts_val, truncation=True, padding=True, return_tensors="pt"
        )
        train_loader = DataLoader(
            TensorDataset(
                tr_enc["input_ids"], tr_enc["attention_mask"], torch.tensor(labels_tr)
            ),
            batch_size=bs,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                va_enc["input_ids"], va_enc["attention_mask"], torch.tensor(labels_val)
            ),
            batch_size=bs,
        )

        # init model and optimizer on device
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        ).to(device)
        optim = Adam(model.parameters(), lr=lr)

        # init logging
        key = f"{name}_{ablation}"
        experiment_data[key] = {
            "losses": {"train": [], "val": []},
            "metrics": {"val": []},
            "predictions": [],
            "ground_truth": labels_val,
        }

        # training & evaluation loop
        for ep in range(1, epochs + 1):
            # train
            model.train()
            tr_losses = []
            for batch in train_loader:
                batch = tuple(
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
                )
                ids, msk, lbl = batch
                optim.zero_grad()
                out = model(input_ids=ids, attention_mask=msk, labels=lbl)
                out.loss.backward()
                optim.step()
                tr_losses.append(out.loss.item())
            lt = float(np.mean(tr_losses))
            experiment_data[key]["losses"]["train"].append({"epoch": ep, "loss": lt})

            # val loss
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = tuple(
                        x.to(device) if isinstance(x, torch.Tensor) else x
                        for x in batch
                    )
                    ids, msk, lbl = batch
                    out = model(input_ids=ids, attention_mask=msk, labels=lbl)
                    val_losses.append(out.loss.item())
            lv = float(np.mean(val_losses))
            experiment_data[key]["losses"]["val"].append({"epoch": ep, "loss": lv})
            print(f"Epoch {ep}: validation_loss = {lv:.4f}")

            # detection metrics
            uncs_v, uncs_k, errs = [], [], []
            for i, txt in enumerate(texts_val):
                variants = [txt] + paras_val[i]
                enc = tokenizer(
                    variants, truncation=True, padding=True, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                probs = torch.softmax(logits, -1).cpu()
                preds = probs.argmax(dim=-1).tolist()
                maj = max(set(preds), key=preds.count)
                uncs_v.append(1 - preds.count(maj) / len(preds))
                kl_list = []
                for a in range(len(probs)):
                    for b in range(a + 1, len(probs)):
                        P, Q = probs[a], probs[b]
                        kl1 = F.kl_div(Q.log(), P, reduction="sum").item()
                        kl2 = F.kl_div(P.log(), Q, reduction="sum").item()
                        kl_list.append(0.5 * (kl1 + kl2))
                uncs_k.append(float(np.mean(kl_list)))
                errs.append(int(preds[0] != int(labels_val[i])))

            # AUC and DES
            try:
                auc_v = roc_auc_score(errs, uncs_v)
            except:
                auc_v = 0.5
            try:
                auc_k = roc_auc_score(errs, uncs_k)
            except:
                auc_k = 0.5
            des_v = auc_v / (K + 1)
            des_k = auc_k / (K + 1)

            # Spearman correlation
            try:
                rho_v, _ = spearmanr(errs, uncs_v)
            except:
                rho_v = 0.0
            try:
                rho_k, _ = spearmanr(errs, uncs_k)
            except:
                rho_k = 0.0

            # log metrics
            experiment_data[key]["metrics"]["val"].append(
                {
                    "epoch": ep,
                    "auc_vote": float(auc_v),
                    "DES_vote": float(des_v),
                    "spearman_vote": float(rho_v),
                    "auc_kl": float(auc_k),
                    "DES_kl": float(des_k),
                    "spearman_kl": float(rho_k),
                }
            )
            print(
                f"Epoch {ep}: AUC_vote={auc_v:.3f}, DES_vote={des_v:.3f}, Spearman_vote={rho_v:.3f}, "
                f"AUC_kl={auc_k:.3f}, DES_kl={des_k:.3f}, Spearman_kl={rho_k:.3f}"
            )

        # final predictions
        preds = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(
                    x.to(device) if isinstance(x, torch.Tensor) else x for x in batch
                )
                ids, msk, _ = batch
                logits = model(input_ids=ids, attention_mask=msk).logits
                preds += logits.argmax(dim=-1).cpu().tolist()
        experiment_data[key]["predictions"] = preds

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
