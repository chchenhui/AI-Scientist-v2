# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}
experiment_data = {}

for name, hf_name in hf_datasets.items():
    ds = load_dataset(hf_name)
    train = ds["train"].shuffle(seed=42).select(range(1000))
    test = ds["test"].shuffle(seed=42).select(range(200))
    text_col = "text" if "text" in train.column_names else "content"
    tr_txt, te_txt = train[text_col], test[text_col]
    y_tr, y_te = train["label"], test["label"]

    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(tr_txt + te_txt)
    X_tr_np = tfidf.transform(tr_txt).toarray()
    X_te_np = tfidf.transform(te_txt).toarray()
    ent_tr = -np.sum(X_tr_np * np.log(X_tr_np + 1e-10), axis=1, keepdims=True)

    X_train = torch.tensor(X_tr_np, dtype=torch.float32)
    ent_train = torch.tensor(ent_tr, dtype=torch.float32)
    y_train_t = torch.tensor(y_tr, dtype=torch.long)
    X_test = torch.tensor(X_te_np, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_te, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train, ent_train, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_dim = X_train.shape[1]
    num_classes = len(set(y_tr))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    class DVN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    main_model = MLP().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
    crit_main = nn.CrossEntropyLoss(reduction="none")
    crit_dvn = nn.MSELoss()

    experiment_data[name] = {
        "val_loss": [],
        "val_acc": [],
        "corrs": [],
        "N_meta_history": [],
    }
    N_meta, prev_corr = 10, None
    K_meta, epochs = 20, 3

    for epoch in range(epochs):
        main_model.train()
        step = 0
        for Xb, entb, yb in train_loader:
            batch = {"X": Xb.to(device), "ent": entb.to(device), "y": yb.to(device)}
            logits = main_model(batch["X"])
            loss_i = crit_main(logits, batch["y"])
            # representation norm feature
            reps = main_model.net[1](main_model.net[0](batch["X"]))
            rep_norm = torch.norm(reps, dim=1, keepdim=True)
            feats = torch.cat(
                [loss_i.detach().unsqueeze(1), batch["ent"], rep_norm], dim=1
            )
            weights = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
            loss = (weights * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()

            if step % N_meta == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = nn.CrossEntropyLoss()(
                        main_model(X_test), y_test_t
                    ).item()
                feats_list, contr_list = [], []
                base_state = main_model.state_dict()
                for idx in random.sample(range(len(X_train)), K_meta):
                    xi = X_train[idx].unsqueeze(0).to(device)
                    yi = y_train_t[idx].unsqueeze(0).to(device)
                    with torch.no_grad():
                        li = crit_main(main_model(xi), yi).item()
                        ent_i = ent_train[idx].item()
                        # compute rep norm for meta feature
                        rep_i = main_model.net[1](main_model.net[0](xi))
                        rep_norm_i = torch.norm(rep_i, dim=1).item()
                    feats_list.append([li, ent_i, rep_norm_i])
                    clone = MLP().to(device)
                    clone.load_state_dict(base_state)
                    opt_c = torch.optim.Adam(clone.parameters(), lr=1e-3)
                    clone.train()
                    lc = crit_main(clone(xi), yi).mean()
                    opt_c.zero_grad()
                    lc.backward()
                    opt_c.step()
                    clone.eval()
                    with torch.no_grad():
                        new_loss = nn.CrossEntropyLoss()(clone(X_test), y_test_t).item()
                    contr_list.append([base_loss - new_loss])
                feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(device)
                contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(device)
                for _ in range(5):
                    dvn_model.train()
                    p = dvn_model(feats_meta)
                    dvn_loss = crit_dvn(p, contr_meta)
                    optimizer_dvn.zero_grad()
                    dvn_loss.backward()
                    optimizer_dvn.step()
                dvn_model.eval()
                with torch.no_grad():
                    preds = dvn_model(feats_meta).cpu().numpy().flatten()
                true = contr_meta.cpu().numpy().flatten()
                corr = spearmanr(preds, true).correlation
                experiment_data[name]["corrs"].append(corr)
                if prev_corr is not None:
                    N_meta = (
                        min(50, N_meta * 2) if corr > prev_corr else max(1, N_meta // 2)
                    )
                experiment_data[name]["N_meta_history"].append(N_meta)
                prev_corr = corr
                print(
                    f"[{name}] Step {step}: Spearman Corr={corr:.4f}, N_meta={N_meta}"
                )
                main_model.train()
            step += 1

        main_model.eval()
        with torch.no_grad():
            logits_val = main_model(X_test)
            val_loss = nn.CrossEntropyLoss()(logits_val, y_test_t).item()
            acc = (logits_val.argmax(1) == y_test_t).float().mean().item()
        experiment_data[name]["val_loss"].append(val_loss)
        experiment_data[name]["val_acc"].append(acc)
        print(f"[{name}] Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={acc:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
