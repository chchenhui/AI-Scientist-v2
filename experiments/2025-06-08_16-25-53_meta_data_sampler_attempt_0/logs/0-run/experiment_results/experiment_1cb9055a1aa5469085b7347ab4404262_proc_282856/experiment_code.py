import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# datasets to run
hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}
# meta‐sample sizes to ablate
K_meta_list = [1, 5, 20, 50, 100]
epochs = 3
batch_size = 64
# container for all results
experiment_data = {}


# define models
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
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


# main ablation loop
for K_meta in K_meta_list:
    key = f"Ablate_Meta_Sample_Size_K={K_meta}"
    experiment_data[key] = {}
    for name, hf_name in hf_datasets.items():
        # load and preprocess
        ds = load_dataset(hf_name)
        train = ds["train"].shuffle(seed=42).select(range(1000))
        test = ds["test"].shuffle(seed=42).select(range(200))
        text_col = "text" if "text" in train.column_names else "content"
        tr_txt, te_txt = train[text_col], test[text_col]
        y_tr, y_te = train["label"], test["label"]
        tfidf = TfidfVectorizer(max_features=500, norm="l2")
        tfidf.fit(tr_txt + te_txt)
        X_tr = tfidf.transform(tr_txt).toarray()
        X_te = tfidf.transform(te_txt).toarray()
        ent_tr = -np.sum(X_tr * np.log(X_tr + 1e-10), axis=1, keepdims=True)
        # tensors
        X_train = torch.tensor(X_tr, dtype=torch.float32)
        ent_train = torch.tensor(ent_tr, dtype=torch.float32)
        y_train = torch.tensor(y_tr, dtype=torch.long)
        X_test = torch.tensor(X_te, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_te, dtype=torch.long).to(device)
        # dataloader
        train_ds = TensorDataset(X_train, ent_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        # model, optimizers, losses
        input_dim = X_train.shape[1]
        num_classes = len(set(y_tr))
        main_model = MLP(input_dim, num_classes).to(device)
        dvn_model = DVN().to(device)
        opt_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
        opt_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
        crit_main = nn.CrossEntropyLoss(reduction="none")
        crit_dvn = nn.MSELoss()
        # trackers
        exp = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "corrs": [],
            "N_meta_history": [],
            "predictions": None,
            "ground_truth": None,
        }
        N_meta = 10
        prev_corr = None
        # training
        for epoch in range(epochs):
            main_model.train()
            step = 0
            for Xb, entb, yb in train_loader:
                Xb, entb, yb = Xb.to(device), entb.to(device), yb.to(device)
                # forward main
                logits = main_model(Xb)
                loss_i = crit_main(logits, yb)  # per‐sample
                # rep norm feature
                reps = main_model.net[1](main_model.net[0](Xb))
                rep_norm = torch.norm(reps, dim=1, keepdim=True)
                feats = torch.cat([loss_i.detach().unsqueeze(1), entb, rep_norm], dim=1)
                w = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
                loss = (w * loss_i).sum()
                # update main
                opt_main.zero_grad()
                loss.backward()
                opt_main.step()
                # meta‐update
                if step % N_meta == 0:
                    main_model.eval()
                    with torch.no_grad():
                        base_loss = nn.CrossEntropyLoss()(
                            main_model(X_test), y_test
                        ).item()
                    feats_list, contr_list = [], []
                    base_state = main_model.state_dict()
                    for idx in random.sample(range(len(X_train)), K_meta):
                        xi = X_train[idx].unsqueeze(0).to(device)
                        yi = y_train[idx].unsqueeze(0).to(device)
                        with torch.no_grad():
                            li = crit_main(main_model(xi), yi).item()
                            ent_i = ent_train[idx].item()
                            rep_i = main_model.net[1](main_model.net[0](xi))
                            rep_norm_i = torch.norm(rep_i, dim=1).item()
                        feats_list.append([li, ent_i, rep_norm_i])
                        clone = MLP(input_dim, num_classes).to(device)
                        clone.load_state_dict(base_state)
                        opt_c = torch.optim.Adam(clone.parameters(), lr=1e-3)
                        clone.train()
                        lc = crit_main(clone(xi), yi).mean()
                        opt_c.zero_grad()
                        lc.backward()
                        opt_c.step()
                        clone.eval()
                        with torch.no_grad():
                            new_loss = nn.CrossEntropyLoss()(
                                clone(X_test), y_test
                            ).item()
                        contr_list.append([base_loss - new_loss])
                    feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(
                        device
                    )
                    contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(
                        device
                    )
                    for _ in range(5):
                        dvn_model.train()
                        dvn_loss = crit_dvn(dvn_model(feats_meta), contr_meta)
                        opt_dvn.zero_grad()
                        dvn_loss.backward()
                        opt_dvn.step()
                    dvn_model.eval()
                    with torch.no_grad():
                        preds = dvn_model(feats_meta).cpu().numpy().flatten()
                    true = contr_meta.cpu().numpy().flatten()
                    corr = spearmanr(preds, true).correlation
                    exp["corrs"].append(corr)
                    if prev_corr is not None:
                        N_meta = (
                            min(50, N_meta * 2)
                            if corr > prev_corr
                            else max(1, N_meta // 2)
                        )
                    exp["N_meta_history"].append(N_meta)
                    prev_corr = corr
                    main_model.train()
                step += 1
            # end epoch: eval train & val
            main_model.eval()
            with torch.no_grad():
                # train metrics
                logits_tr = main_model(X_train.to(device))
                tr_loss = nn.CrossEntropyLoss()(logits_tr, y_train.to(device)).item()
                tr_acc = (
                    (logits_tr.argmax(1) == y_train.to(device)).float().mean().item()
                )
                # val metrics
                logits_va = main_model(X_test)
                va_loss = nn.CrossEntropyLoss()(logits_va, y_test).item()
                va_acc = (logits_va.argmax(1) == y_test).float().mean().item()
            exp["losses"]["train"].append(tr_loss)
            exp["metrics"]["train"].append(tr_acc)
            exp["losses"]["val"].append(va_loss)
            exp["metrics"]["val"].append(va_acc)
            print(
                f"[{key}][{name}] Epoch {epoch}: tr_loss={tr_loss:.4f}, tr_acc={tr_acc:.4f}, val_loss={va_loss:.4f}, val_acc={va_acc:.4f}"
            )
        # final test predictions
        main_model.eval()
        with torch.no_grad():
            final_preds = main_model(X_test).argmax(1).cpu().numpy()
        exp["predictions"] = final_preds
        exp["ground_truth"] = y_test.cpu().numpy()
        # store
        experiment_data[key][name] = exp

# save all data
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
