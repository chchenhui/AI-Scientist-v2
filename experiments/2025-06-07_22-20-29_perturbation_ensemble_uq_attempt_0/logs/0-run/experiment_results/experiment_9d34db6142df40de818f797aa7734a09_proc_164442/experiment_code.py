import os, random, re, nltk
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# download WordNet
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet


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


# datasets to run
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}

# hyperparameters
train_size, val_size = 5000, 500
K, epochs, bs, lr = 5, 5, 32, 2e-5

# tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# main experiment data container
experiment_data = {"No_Dropout_Ablation": {}}

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

for ds_name, (ds, sub, text_col, label_col) in datasets_info.items():
    # load and sample
    train_split = (
        load_dataset(ds, sub, split="train") if sub else load_dataset(ds, split="train")
    )
    val_split = load_dataset(ds, sub, split="validation" if sub else "test")
    train_split = train_split.shuffle(42).select(range(train_size))
    val_split = val_split.shuffle(42).select(range(val_size))
    texts_train, labels_train = train_split[text_col], train_split[label_col]
    texts_val, labels_val = val_split[text_col], val_split[label_col]
    # generate paraphrases
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

    # init model and disable dropout
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    optimizer = Adam(model.parameters(), lr=lr)

    # record structures
    expd = {
        "losses": {"train": [], "val": []},
        "metrics": {"detection": []},
        "predictions": [],
        "ground_truth": labels_val,
    }
    experiment_data["No_Dropout_Ablation"][ds_name] = expd

    # training + validation + detection
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for ids, mask, lbl in tr_loader:
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=lbl)
            out.loss.backward()
            optimizer.step()
            train_losses.append(out.loss.item())
        expd["losses"]["train"].append(
            {"epoch": epoch, "loss": float(np.mean(train_losses))}
        )

        # validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for ids, mask, lbl in va_loader:
                ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
                out = model(input_ids=ids, attention_mask=mask, labels=lbl)
                val_losses.append(out.loss.item())
        vl = float(np.mean(val_losses))
        expd["losses"]["val"].append({"epoch": epoch, "loss": vl})
        print(f"{ds_name} Epoch {epoch}: val_loss={vl:.4f}")

        # detection metrics
        uncs_vote, uncs_kl, errs = [], [], []
        for i, txt in enumerate(texts_val):
            probs, preds = [], []
            for variant in [txt] + paras[i]:
                enc = tokenizer(
                    variant, truncation=True, padding=True, return_tensors="pt"
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
            errs.append(int(preds[0] != int(labels_val[i])))
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
        expd["metrics"]["detection"].append(
            {
                "epoch": epoch,
                "auc_vote": auc_v,
                "DES_vote": des_v,
                "auc_kl": auc_k,
                "DES_kl": des_k,
            }
        )
        print(
            f"{ds_name} Epoch {epoch}: AUC_vote={auc_v:.4f}, DES_vote={des_v:.4f}, AUC_kl={auc_k:.4f}, DES_kl={des_k:.4f}"
        )

    # final predictions
    preds = []
    model.eval()
    with torch.no_grad():
        for ids, mask, _ in va_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(torch.argmax(logits, -1).cpu().tolist())
    expd["predictions"] = preds

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
