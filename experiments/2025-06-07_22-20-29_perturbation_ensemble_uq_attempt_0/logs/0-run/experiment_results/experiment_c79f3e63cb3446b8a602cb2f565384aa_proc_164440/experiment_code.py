import os, random, re
import nltk, torch, numpy as np
import torch.nn.functional as F
from nltk.corpus import wordnet
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

# setup
nltk.download("wordnet", quiet=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def prune_heads(model, prune_ratio=0.5, seed=42):
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = cfg.hidden_size // H
    random.seed(seed)
    # build head_mask and record per-layer head configs
    head_mask = torch.ones(L, H, dtype=torch.float32)
    layer_masks = {}
    for l in range(L):
        pruned = set(random.sample(range(H), int(H * prune_ratio)))
        m = [0.0 if h in pruned else 1.0 for h in range(H)]
        head_mask[l] = torch.tensor(m)
        layer_masks[l] = m
    # attach gradient hooks to freeze pruned subspaces
    for l, mask in layer_masks.items():
        attn_mod = model.bert.encoder.layer[l].attention
        # query/key/value
        for proj in (attn_mod.self.query, attn_mod.self.key, attn_mod.self.value):
            wm = torch.repeat_interleave(
                torch.tensor(mask, dtype=torch.float32), head_dim
            )
            wm2d = wm.unsqueeze(1).to(proj.weight.device).expand_as(proj.weight)
            bm = wm.to(proj.bias.device)
            proj.weight.register_hook(lambda grad, m=wm2d: grad * m)
            proj.bias.register_hook(lambda grad, m=bm: grad * m)
        # output projection
        dense = attn_mod.output.dense
        om = torch.ones_like(dense.weight)
        for h, keep in enumerate(mask):
            if keep == 0.0:
                om[:, h * head_dim : (h + 1) * head_dim] = 0.0
        om = om.to(dense.weight.device)
        dense.weight.register_hook(lambda grad, m=om: grad * m)
    return head_mask.to(device)


# configs
datasets_info = {
    "sst2": ("glue", "sst2", "sentence", "label"),
    "yelp_polarity": ("yelp_polarity", None, "text", "label"),
    "imdb": ("imdb", None, "text", "label"),
}
train_size, val_size = 5000, 500
K, epochs, bs, lr = 5, 5, 32, 2e-5

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# preprocess data once
data = {}
for name, (ds, sub, text_col, label_col) in datasets_info.items():
    dstr = (
        load_dataset(ds, sub, split="train") if sub else load_dataset(ds, split="train")
    )
    dsvl = load_dataset(ds, sub, split="validation" if sub else "test")
    dstr = dstr.shuffle(42).select(range(train_size))
    dsvl = dsvl.shuffle(42).select(range(val_size))
    texts_tr, labs_tr = dstr[text_col], dstr[label_col]
    texts_va, labs_va = dsvl[text_col], dsvl[label_col]
    paras = {i: generate_paraphrases(texts_va[i], K) for i in range(len(texts_va))}
    tr_enc = tokenizer(texts_tr, truncation=True, padding=True, return_tensors="pt")
    va_enc = tokenizer(texts_va, truncation=True, padding=True, return_tensors="pt")
    tr_ds = TensorDataset(
        tr_enc["input_ids"], tr_enc["attention_mask"], torch.tensor(labs_tr)
    )
    va_ds = TensorDataset(
        va_enc["input_ids"], va_enc["attention_mask"], torch.tensor(labs_va)
    )
    data[name] = {
        "tr_loader": DataLoader(tr_ds, batch_size=bs, shuffle=True),
        "va_loader": DataLoader(va_ds, batch_size=bs),
        "texts_val": texts_va,
        "labels_val": labs_va,
        "paras": paras,
        "train_size": len(texts_tr),
        "val_size": len(texts_va),
    }

# run experiments
experiment_data = {"full_heads": {}, "pruned_heads": {}}
for exp in experiment_data:
    for name in datasets_info:
        dd = data[name]
        # init model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        ).to(device)
        head_mask = None
        if exp == "pruned_heads":
            head_mask = prune_heads(model)
        optimizer = Adam(model.parameters(), lr=lr)
        # prepare logging
        experiment_data[exp][name] = {
            "losses": {"train": [], "val": []},
            "metrics": {"train": [], "val": []},
            "detection": [],
            "predictions": [],
            "ground_truth": dd["labels_val"],
        }
        # epochs
        for epoch in range(1, epochs + 1):
            # train
            model.train()
            train_loss, train_corr = 0.0, 0
            for ids, mask, labs in dd["tr_loader"]:
                ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
                optimizer.zero_grad()
                kwargs = {"input_ids": ids, "attention_mask": mask}
                if head_mask is not None:
                    kwargs["head_mask"] = head_mask
                out = model(**kwargs, labels=labs)
                out.loss.backward()
                optimizer.step()
                train_loss += out.loss.item()
                preds = out.logits.argmax(dim=-1)
                train_corr += (preds == labs).sum().item()
            avg_tr_loss = train_loss / len(dd["tr_loader"])
            tr_acc = train_corr / dd["train_size"]
            experiment_data[exp][name]["losses"]["train"].append(
                {"epoch": epoch, "loss": avg_tr_loss}
            )
            experiment_data[exp][name]["metrics"]["train"].append(
                {"epoch": epoch, "acc": tr_acc}
            )

            # validation
            model.eval()
            val_loss, val_corr = 0.0, 0
            with torch.no_grad():
                for ids, mask, labs in dd["va_loader"]:
                    ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
                    kwargs = {"input_ids": ids, "attention_mask": mask}
                    if head_mask is not None:
                        kwargs["head_mask"] = head_mask
                    out = model(**kwargs, labels=labs)
                    val_loss += out.loss.item()
                    preds = out.logits.argmax(dim=-1)
                    val_corr += (preds == labs).sum().item()
            avg_va_loss = val_loss / len(dd["va_loader"])
            va_acc = val_corr / dd["val_size"]
            experiment_data[exp][name]["losses"]["val"].append(
                {"epoch": epoch, "loss": avg_va_loss}
            )
            experiment_data[exp][name]["metrics"]["val"].append(
                {"epoch": epoch, "acc": va_acc}
            )
            print(
                f"{exp} {name} E{epoch}: tr_loss={avg_tr_loss:.4f}, tr_acc={tr_acc:.4f}, "
                f"va_loss={avg_va_loss:.4f}, va_acc={va_acc:.4f}"
            )

            # detection metrics
            uncs_vote, uncs_kl, errs = [], [], []
            for i, txt in enumerate(dd["texts_val"]):
                probs, ppreds = [], []
                for var in [txt] + dd["paras"][i]:
                    enc = tokenizer(
                        var, return_tensors="pt", truncation=True, padding=True
                    ).to(device)
                    kwargs = {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                    }
                    if head_mask is not None:
                        kwargs["head_mask"] = head_mask
                    with torch.no_grad():
                        lg = model(**kwargs).logits
                    p = F.softmax(lg, dim=-1).squeeze(0).cpu()
                    probs.append(p)
                    ppreds.append(int(p.argmax().item()))
                maj = max(set(ppreds), key=ppreds.count)
                uncs_vote.append(1.0 - ppreds.count(maj) / len(ppreds))
                klv = []
                for a in range(len(probs)):
                    for b in range(a + 1, len(probs)):
                        P, Q = probs[a], probs[b]
                        kl1 = F.kl_div(Q.log(), P, reduction="sum").item()
                        kl2 = F.kl_div(P.log(), Q, reduction="sum").item()
                        klv.append(0.5 * (kl1 + kl2))
                uncs_kl.append(float(np.mean(klv)))
                errs.append(int(ppreds[0] != int(dd["labels_val"][i])))
            try:
                auc_v = roc_auc_score(errs, uncs_vote)
            except:
                auc_v = 0.5
            try:
                auc_k = roc_auc_score(errs, uncs_kl)
            except:
                auc_k = 0.5
            des_v, des_k = auc_v / (K + 1), auc_k / (K + 1)
            experiment_data[exp][name]["detection"].append(
                {
                    "epoch": epoch,
                    "auc_vote": auc_v,
                    "DES_vote": des_v,
                    "auc_kl": auc_k,
                    "DES_kl": des_k,
                }
            )
            print(
                f"{exp} {name} E{epoch}: AUC_vote={auc_v:.4f}, DES_vote={des_v:.4f}, "
                f"AUC_kl={auc_k:.4f}, DES_kl={des_k:.4f}"
            )

        # final predictions
        preds = []
        model.eval()
        with torch.no_grad():
            for ids, mask, _ in dd["va_loader"]:
                ids, mask = ids.to(device), mask.to(device)
                kwargs = {"input_ids": ids, "attention_mask": mask}
                if head_mask is not None:
                    kwargs["head_mask"] = head_mask
                lg = model(**kwargs).logits
                preds.extend(lg.argmax(dim=-1).cpu().tolist())
        experiment_data[exp][name]["predictions"] = preds

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy in", working_dir)
