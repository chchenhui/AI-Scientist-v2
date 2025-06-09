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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download("wordnet", quiet=True)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


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


datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}

# hyperparameters
train_size, val_size = 5000, 500
K, epochs, bs, lr = 5, 5, 32, 2e-5

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
experiment_data = {"positional_embedding_ablation": {}}

for name, (ds, sub, text_col, label_col) in datasets_info.items():
    # load and sample
    ds_train = (
        load_dataset(ds, sub, split="train") if sub else load_dataset(ds, split="train")
    )
    ds_val = load_dataset(ds, sub, split="validation" if sub else "test")
    ds_train = ds_train.shuffle(42).select(range(train_size))
    ds_val = ds_val.shuffle(42).select(range(val_size))
    texts_train, labels_train = ds_train[text_col], ds_train[label_col]
    texts_val, labels_val = ds_val[text_col], ds_val[label_col]
    paras = {i: generate_paraphrases(t, K) for i, t in enumerate(texts_val)}

    # tokenize & dataloaders
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

    # model + ablation
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)
    # zero + freeze positional embeddings
    model.bert.embeddings.position_embeddings.weight.data.zero_()
    model.bert.embeddings.position_embeddings.weight.requires_grad = False
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    # storage
    data = {
        "losses": {"train": [], "val": []},
        "metrics": {"train": [], "val": []},
        "detection": [],
        "predictions": [],
        "ground_truth": labels_val,
    }

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        train_losses = []
        for ids, mask, lbl in tr_loader:
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=lbl)
            out.loss.backward()
            optimizer.step()
            train_losses.append(out.loss.item())
        train_loss = float(np.mean(train_losses))
        data["losses"]["train"].append(train_loss)

        # validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for ids, mask, lbl in va_loader:
                ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
                out = model(input_ids=ids, attention_mask=mask, labels=lbl)
                val_losses.append(out.loss.item())
        val_loss = float(np.mean(val_losses))
        data["losses"]["val"].append(val_loss)

        # classification accuracies
        correct_tr, total_tr = 0, 0
        with torch.no_grad():
            for ids, mask, lbl in tr_loader:
                ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                preds = logits.argmax(-1)
                correct_tr += (preds == lbl).sum().item()
                total_tr += lbl.size(0)
        tr_acc = correct_tr / total_tr
        correct_va, total_va = 0, 0
        with torch.no_grad():
            for ids, mask, lbl in va_loader:
                ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                preds = logits.argmax(-1)
                correct_va += (preds == lbl).sum().item()
                total_va += lbl.size(0)
        va_acc = correct_va / total_va
        data["metrics"]["train"].append(tr_acc)
        data["metrics"]["val"].append(va_acc)

        # paraphrase‐based detection
        uncs_vote, uncs_kl, errs = [], [], []
        for i, txt in enumerate(texts_val):
            probs, preds = [], []
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
            kl_vals = []
            for a in range(len(probs)):
                for b in range(a + 1, len(probs)):
                    P, Q = probs[a], probs[b]
                    kl_vals.append(
                        0.5
                        * (
                            F.kl_div(Q.log(), P, reduction="sum").item()
                            + F.kl_div(P.log(), Q, reduction="sum").item()
                        )
                    )
            uncs_kl.append(float(np.mean(kl_vals)))
            errs.append(int(preds[0] != int(labels_val[i])))
        try:
            auc_v = roc_auc_score(errs, uncs_vote)
        except:
            auc_v = 0.5
        try:
            auc_k = roc_auc_score(errs, uncs_kl)
        except:
            auc_k = 0.5
        des_v, des_k = auc_v / (K + 1), auc_k / (K + 1)
        data["detection"].append(
            {
                "epoch": epoch,
                "auc_vote": auc_v,
                "DES_vote": des_v,
                "auc_kl": auc_k,
                "DES_kl": des_k,
            }
        )

        print(
            f"{name} Epoch {epoch} | tr_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"tr_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
            f"AUC_vote={auc_v:.4f} DES_vote={des_v:.4f} AUC_kl={auc_k:.4f} DES_kl={des_k:.4f}"
        )

    # final predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, mask, _ in va_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
    data["predictions"] = preds

    experiment_data["positional_embedding_ablation"][name] = data

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
