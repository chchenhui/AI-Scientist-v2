# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os, random, re
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

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# datasets and hyperparams
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}
train_size, val_size = 5000, 500
K, epochs, bs, lr = 5, 5, 32, 2e-5
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ablation settings
ablations = {"baseline": False, "RandomInitEmbeddingAblation": True}
experiment_data = {}

for ablation_name, do_reinit in ablations.items():
    experiment_data[ablation_name] = {}
    for name, (ds, sub, text_col, label_col) in datasets_info.items():
        # load & subsample
        ds_train = (
            load_dataset(ds, sub, split="train")
            if sub
            else load_dataset(ds, split="train")
        )
        ds_val = load_dataset(ds, sub, split="validation" if sub else "test")
        ds_train = ds_train.shuffle(42).select(range(train_size))
        ds_val = ds_val.shuffle(42).select(range(val_size))
        texts_train, labels_train = ds_train[text_col], ds_train[label_col]
        texts_val, labels_val = ds_val[text_col], ds_val[label_col]
        paras = {i: generate_paraphrases(t, K) for i, t in enumerate(texts_val)}
        # tokenize
        tr_enc = tokenizer(
            texts_train, truncation=True, padding=True, return_tensors="pt"
        )
        va_enc = tokenizer(
            texts_val, truncation=True, padding=True, return_tensors="pt"
        )
        train_ds = TensorDataset(
            tr_enc["input_ids"], tr_enc["attention_mask"], torch.tensor(labels_train)
        )
        val_ds = TensorDataset(
            va_enc["input_ids"], va_enc["attention_mask"], torch.tensor(labels_val)
        )
        tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        va_loader = DataLoader(val_ds, batch_size=bs)
        # init storage
        experiment_data[ablation_name][name] = {
            "losses": {"train": [], "val": []},
            "metrics": {"detection": []},
            "predictions": [],
            "ground_truth": labels_val.copy(),
        }
        # model & ablation
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        ).to(device)
        if do_reinit:
            emb = model.get_input_embeddings()
            torch.nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        optimizer = Adam(model.parameters(), lr=lr)
        # train/eval
        for epoch in range(1, epochs + 1):
            model.train()
            tr_losses = []
            for ids, mask, lbls in tr_loader:
                ids, mask, lbls = ids.to(device), mask.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                out.loss.backward()
                optimizer.step()
                tr_losses.append(out.loss.item())
            experiment_data[ablation_name][name]["losses"]["train"].append(
                {"epoch": epoch, "loss": float(np.mean(tr_losses))}
            )
            model.eval()
            va_losses = []
            with torch.no_grad():
                for ids, mask, lbls in va_loader:
                    ids, mask, lbls = ids.to(device), mask.to(device), lbls.to(device)
                    out = model(input_ids=ids, attention_mask=mask, labels=lbls)
                    va_losses.append(out.loss.item())
            val_loss = float(np.mean(va_losses))
            experiment_data[ablation_name][name]["losses"]["val"].append(
                {"epoch": epoch, "loss": val_loss}
            )
            # detection metrics
            uncs_vote, uncs_kl, errs = [], [], []
            for i, txt in enumerate(texts_val):
                probs, preds = [], []
                for var in [txt] + paras[i]:
                    enc = tokenizer(
                        var, return_tensors="pt", truncation=True, padding=True
                    ).to(device)
                    with torch.no_grad():
                        logits = model(**enc).logits
                    p = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                    probs.append(p)
                    preds.append(int(p.argmax().item()))
                maj = max(set(preds), key=preds.count)
                uncs_vote.append(1 - preds.count(maj) / len(preds))
                kl_vals = []
                for a in range(len(probs)):
                    for b in range(a + 1, len(probs)):
                        P, Q = probs[a], probs[b]
                        kl1 = F.kl_div(Q.log(), P, reduction="sum").item()
                        kl2 = F.kl_div(P.log(), Q, reduction="sum").item()
                        kl_vals.append(0.5 * (kl1 + kl2))
                uncs_kl.append(float(np.mean(kl_vals)))
                errs.append(int(preds[0] != labels_val[i]))
            auc_v = roc_auc_score(errs, uncs_vote) if len(set(errs)) > 1 else 0.5
            auc_k = roc_auc_score(errs, uncs_kl) if len(set(errs)) > 1 else 0.5
            des_v = auc_v / (K + 1)
            des_k = auc_k / (K + 1)
            experiment_data[ablation_name][name]["metrics"]["detection"].append(
                {
                    "epoch": epoch,
                    "auc_vote": auc_v,
                    "DES_vote": des_v,
                    "auc_kl": auc_k,
                    "DES_kl": des_k,
                }
            )
        # final predictions
        preds = []
        model.eval()
        with torch.no_grad():
            for ids, mask, _ in va_loader:
                ids, mask = ids.to(device), mask.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                preds.extend(torch.argmax(logits, -1).cpu().tolist())
        experiment_data[ablation_name][name]["predictions"] = preds

# save data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
