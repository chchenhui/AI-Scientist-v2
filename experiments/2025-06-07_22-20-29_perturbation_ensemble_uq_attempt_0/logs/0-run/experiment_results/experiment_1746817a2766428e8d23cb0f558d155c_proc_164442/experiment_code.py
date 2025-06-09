import os, random, re
import nltk
import torch
import numpy as np
import torch.nn.functional as F
from nltk.corpus import wordnet
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
nltk.download("wordnet", quiet=True)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ablation
ABL = "No_Pretraining_RandomInit"
experiment_data = {ABL: {}}


# paraphrase utility
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


# settings
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}
train_size, val_size = 5000, 500
K, epochs, bs, lr = 5, 5, 32, 2e-5
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

for name, (ds, sub, text_col, label_col) in datasets_info.items():
    # load data
    ds_tr = (
        load_dataset(ds, sub, split="train") if sub else load_dataset(ds, split="train")
    )
    ds_va = load_dataset(ds, sub, split="validation" if sub else "test")
    ds_tr = ds_tr.shuffle(42).select(range(train_size))
    ds_va = ds_va.shuffle(42).select(range(val_size))
    texts_tr, labels_tr = ds_tr[text_col], ds_tr[label_col]
    texts_va, labels_va = ds_va[text_col], ds_va[label_col]
    # paraphrases
    paras = {i: generate_paraphrases(t, K) for i, t in enumerate(texts_va)}

    # tokenize
    tr_enc = tokenizer(texts_tr, truncation=True, padding=True, return_tensors="pt")
    va_enc = tokenizer(texts_va, truncation=True, padding=True, return_tensors="pt")
    train_ds = TensorDataset(
        tr_enc["input_ids"], tr_enc["attention_mask"], torch.tensor(labels_tr)
    )
    val_ds = TensorDataset(
        va_enc["input_ids"], va_enc["attention_mask"], torch.tensor(labels_va)
    )
    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=bs)

    # init random‐weight BERT
    config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
    model = BertForSequenceClassification(config).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # prepare storage
    experiment_data[ABL][name] = {
        "losses": {"train": [], "val": []},
        "metrics": {"detection": []},
        "predictions": [],
        "ground_truth": labels_va,
    }

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        tr_ls = []
        for ids, mask, labs in tr_loader:
            ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            out.loss.backward()
            optimizer.step()
            tr_ls.append(out.loss.item())
        experiment_data[ABL][name]["losses"]["train"].append(
            {"epoch": epoch, "loss": float(np.mean(tr_ls))}
        )

        model.eval()
        va_ls = []
        with torch.no_grad():
            for ids, mask, labs in va_loader:
                ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                va_ls.append(out.loss.item())
        experiment_data[ABL][name]["losses"]["val"].append(
            {"epoch": epoch, "loss": float(np.mean(va_ls))}
        )
        print(f"{name} Epoch {epoch}: val_loss={np.mean(va_ls):.4f}")

        # detection metrics
        uncs_vote, uncs_kl, errs = [], [], []
        for i, txt in enumerate(texts_va):
            probs, preds = [], []
            for var in [txt] + paras[i]:
                enc = tokenizer(
                    var, return_tensors="pt", truncation=True, padding=True
                ).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                p = torch.softmax(logits, -1).squeeze(0).cpu()
                probs.append(p)
                preds.append(int(p.argmax().item()))
            maj = max(set(preds), key=preds.count)
            uncs_vote.append(1 - preds.count(maj) / len(preds))
            klv = []
            for a in range(len(probs)):
                for b in range(a + 1, len(probs)):
                    P, Q = probs[a], probs[b]
                    klv.append(
                        0.5
                        * (
                            F.kl_div(Q.log(), P, reduction="sum").item()
                            + F.kl_div(P.log(), Q, reduction="sum").item()
                        )
                    )
            uncs_kl.append(float(np.mean(klv)))
            errs.append(int(preds[0] != int(labels_va[i])))
        try:
            auc_v = roc_auc_score(errs, uncs_vote)
            auc_k = roc_auc_score(errs, uncs_kl)
        except:
            auc_v = auc_k = 0.5
        des_v = auc_v / (K + 1)
        des_k = auc_k / (K + 1)
        experiment_data[ABL][name]["metrics"]["detection"].append(
            {
                "epoch": epoch,
                "auc_vote": auc_v,
                "DES_vote": des_v,
                "auc_kl": auc_k,
                "DES_kl": des_k,
            }
        )
        print(
            f"{name} Epoch {epoch}: AUC_v={auc_v:.3f}, DES_v={des_v:.3f}, AUC_k={auc_k:.3f}, DES_k={des_k:.3f}"
        )

    # final preds
    preds = []
    model.eval()
    with torch.no_grad():
        for ids, mask, _ in va_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
    experiment_data[ABL][name]["predictions"] = preds

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
