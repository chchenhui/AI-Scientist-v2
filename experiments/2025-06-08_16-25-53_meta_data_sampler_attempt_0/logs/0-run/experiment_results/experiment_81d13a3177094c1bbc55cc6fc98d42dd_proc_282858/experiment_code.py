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
experiment_data = {"Ablate_Meta_Loss_Ranking": {}}

for name, hf_name in hf_datasets.items():
    print(f"Running dataset: {name}")
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

    X_train_dev = X_train.to(device)
    y_train_dev = y_train_t.to(device)

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
    crit_eval = nn.CrossEntropyLoss()
    mr_loss = nn.MarginRankingLoss(margin=1.0)

    experiment_data["Ablate_Meta_Loss_Ranking"][name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": np.array(y_te),
        "corrs": [],
        "N_meta_history": [],
    }

    N_meta, prev_corr = 10, None
    K_meta, epochs = 20, 3

    for epoch in range(epochs):
        main_model.train()
        step = 0
        for Xb, entb, yb in train_loader:
            Xb, entb, yb = Xb.to(device), entb.to(device), yb.to(device)
            logits = main_model(Xb)
            loss_i = crit_main(logits, yb)
            reps = main_model.net[1](main_model.net[0](Xb))
            rep_norm = torch.norm(reps, dim=1, keepdim=True)
            feats = torch.cat([loss_i.detach().unsqueeze(1), entb, rep_norm], dim=1)
            weights = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
            loss = (weights * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()

            if step % N_meta == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = crit_eval(main_model(X_test), y_test_t).item()
                feats_list, contr_list = [], []
                base_state = main_model.state_dict()
                for idx in random.sample(range(len(X_train)), K_meta):
                    xi = X_train[idx].unsqueeze(0).to(device)
                    yi = y_train_t[idx].unsqueeze(0).to(device)
                    with torch.no_grad():
                        li = crit_main(main_model(xi), yi).item()
                        ent_i = ent_train[idx].item()
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
                        new_loss = crit_eval(clone(X_test), y_test_t).item()
                    contr_list.append([base_loss - new_loss])
                feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(device)
                contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(device)

                for _ in range(5):
                    dvn_model.train()
                    preds = dvn_model(feats_meta).squeeze(1)
                    targets = contr_meta.squeeze(1)
                    i_idx, j_idx = torch.triu_indices(K_meta, K_meta, offset=1)
                    pi, pj = preds[i_idx], preds[j_idx]
                    ti, tj = targets[i_idx], targets[j_idx]
                    y = torch.sign(ti - tj)
                    mask = y != 0
                    if mask.any():
                        y, pi, pj = y[mask], pi[mask], pj[mask]
                        dvn_loss = mr_loss(pi, pj, y)
                        optimizer_dvn.zero_grad()
                        dvn_loss.backward()
                        optimizer_dvn.step()

                dvn_model.eval()
                with torch.no_grad():
                    p = dvn_model(feats_meta).cpu().numpy().flatten()
                true = contr_meta.cpu().numpy().flatten()
                corr = spearmanr(p, true).correlation
                exp_ds = experiment_data["Ablate_Meta_Loss_Ranking"][name]
                exp_ds["corrs"].append(corr)
                if prev_corr is not None:
                    N_meta = (
                        min(50, N_meta * 2) if corr > prev_corr else max(1, N_meta // 2)
                    )
                exp_ds["N_meta_history"].append(N_meta)
                prev_corr = corr
                print(
                    f"[{name}] Step {step}: Spearman Corr={corr:.4f}, N_meta={N_meta}"
                )
                main_model.train()
            step += 1

        main_model.eval()
        with torch.no_grad():
            logits_train = main_model(X_train_dev)
            train_loss = crit_eval(logits_train, y_train_dev).item()
            train_acc = (logits_train.argmax(1) == y_train_dev).float().mean().item()
            logits_val = main_model(X_test)
            val_loss = crit_eval(logits_val, y_test_t).item()
            val_acc = (logits_val.argmax(1) == y_test_t).float().mean().item()
        exp_ds = experiment_data["Ablate_Meta_Loss_Ranking"][name]
        exp_ds["losses"]["train"].append(train_loss)
        exp_ds["metrics"]["train"].append(train_acc)
        exp_ds["losses"]["val"].append(val_loss)
        exp_ds["metrics"]["val"].append(val_acc)
        exp_ds["predictions"].append(logits_val.argmax(1).cpu().numpy())
        print(
            f"[{name}] Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

# convert lists to numpy arrays
for ab_type, ds_dict in experiment_data.items():
    for ds_name, data in ds_dict.items():
        data["metrics"]["train"] = np.array(data["metrics"]["train"])
        data["metrics"]["val"] = np.array(data["metrics"]["val"])
        data["losses"]["train"] = np.array(data["losses"]["train"])
        data["losses"]["val"] = np.array(data["losses"]["val"])
        data["predictions"] = np.stack(data["predictions"])
        data["ground_truth"] = np.array(data["ground_truth"])
        data["corrs"] = np.array(data["corrs"])
        data["N_meta_history"] = np.array(data["N_meta_history"])

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
