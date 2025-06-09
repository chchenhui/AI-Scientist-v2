import os
import random
import re
import nltk
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score

# Setup working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Prepare WordNet
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet


def generate_perturbations(text, K):
    words = text.split()
    perturbs = []
    half = K // 2
    # synonym replacements
    for _ in range(half):
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
        perturbs.append(" ".join(new))
    # random swaps
    for _ in range(K - half):
        new = words.copy()
        if len(new) > 1:
            i, j = random.sample(range(len(new)), 2)
            new[i], new[j] = new[j], new[i]
        perturbs.append(" ".join(new))
    return perturbs


# Datasets
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
K, epochs, bs, lr = 5, 4, 32, 3e-5
experiment_data = {}

for name, (ds, sub, text_col, label_col) in datasets_info.items():
    # load and sample
    if sub:
        dset_tr = load_dataset(ds, sub, split="train").shuffle(42).select(range(3000))
        dset_va = (
            load_dataset(ds, sub, split="validation").shuffle(42).select(range(600))
        )
    else:
        dset_tr = load_dataset(ds, split="train").shuffle(42).select(range(3000))
        dset_va = load_dataset(ds, split="test").shuffle(42).select(range(600))
    train_texts = dset_tr[text_col]
    train_labels = dset_tr[label_col]
    val_texts = dset_va[text_col]
    val_labels = dset_va[label_col]
    # precompute perturbations
    paraphrases = {i: generate_perturbations(t, K) for i, t in enumerate(val_texts)}
    # tokenize
    tr_enc = tokenizer(
        train_texts, truncation=True, padding="max_length", max_length=128
    )
    va_enc = tokenizer(val_texts, truncation=True, padding="max_length", max_length=128)
    train_ds = TensorDataset(
        torch.tensor(tr_enc["input_ids"]),
        torch.tensor(tr_enc["attention_mask"]),
        torch.tensor(train_labels),
    )
    val_ds = TensorDataset(
        torch.tensor(va_enc["input_ids"]),
        torch.tensor(va_enc["attention_mask"]),
        torch.tensor(val_labels),
    )
    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=bs)
    # model & optimizer
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    experiment_data[name] = {
        "losses": {"train": [], "val": []},
        "metrics": {"detection": []},
        "predictions": [],
        "ground_truth": list(val_labels),
    }
    # training + detection
    for epoch in range(1, epochs + 1):
        model.train()
        t_losses = []
        for ids, mask, labels in tr_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            t_losses.append(loss.item())
        train_loss = float(np.mean(t_losses))
        experiment_data[name]["losses"]["train"].append(
            {"epoch": epoch, "loss": train_loss}
        )
        # validation loss
        model.eval()
        v_losses = []
        with torch.no_grad():
            for ids, mask, labels in va_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                v_losses.append(out.loss.item())
        val_loss = float(np.mean(v_losses))
        experiment_data[name]["losses"]["val"].append(
            {"epoch": epoch, "loss": val_loss}
        )
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")
        # detection via PIU composite score
        uncs, errs = [], []
        for i, (txt, gt) in enumerate(zip(val_texts, val_labels)):
            preds, probs = [], []
            for alt in [txt] + paraphrases[i]:
                enc = tokenizer(
                    alt,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                ).to(device)
                with torch.no_grad():
                    logits = model(**enc).logits
                p = torch.softmax(logits, dim=-1)[0, 1].item()
                probs.append(p)
                preds.append(int(logits.argmax(-1).item()))
            maj = max(set(preds), key=preds.count)
            drate = 1 - preds.count(maj) / len(preds)
            std_p = float(np.std(probs))
            norm_std = std_p / 0.5
            unc = 0.5 * drate + 0.5 * norm_std
            uncs.append(unc)
            errs.append(int(preds[0] != gt))
        try:
            auc = roc_auc_score(errs, uncs)
        except:
            auc = 0.5
        des = auc / (K + 1)
        experiment_data[name]["metrics"]["detection"].append(
            {"epoch": epoch, "auc": auc, "DES": des}
        )
        print(f"Epoch {epoch}: detection_auc = {auc:.4f}, DES = {des:.4f}")
    # final predictions
    model.eval()
    final_preds = []
    with torch.no_grad():
        for ids, mask, _ in va_loader:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            final_preds.extend(torch.argmax(logits, -1).cpu().tolist())
    experiment_data[name]["predictions"] = final_preds

# persist data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
