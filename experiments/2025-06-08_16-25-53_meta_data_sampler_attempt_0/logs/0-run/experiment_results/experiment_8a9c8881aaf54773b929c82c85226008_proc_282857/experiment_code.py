import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}
ablation_types = ["random_sampling", "hard_sampling"]
experiment_data = {}

for ablation in ablation_types:
    experiment_data[ablation] = {}
    for name, hf_name in hf_datasets.items():
        # load & preprocess
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

        # tensors
        X_train = torch.tensor(X_tr_np, dtype=torch.float32)
        ent_train = torch.tensor(ent_tr, dtype=torch.float32)
        y_train = torch.tensor(y_tr, dtype=torch.long)
        X_test = torch.tensor(X_te_np, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_te, dtype=torch.long).to(device)

        # prepare storage
        ds_data = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": y_te,
            "corrs": [],
            "N_meta_history": [],
        }

        train_ds = TensorDataset(X_train, ent_train, y_train)
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
        ce_full = nn.CrossEntropyLoss()

        N_meta, prev_corr = 10, None
        K_meta, epochs = 20, 3

        for epoch in range(epochs):
            main_model.train()
            step = 0
            for Xb, entb, yb in train_loader:
                Xb, entb, yb = Xb.to(device), entb.to(device), yb.to(device)
                logits = main_model(Xb)
                loss_i = crit_main(logits, yb)
                # rep norm
                reps = main_model.net[1](main_model.net[0](Xb))
                rep_norm = torch.norm(reps, dim=1, keepdim=True)
                feats = torch.cat([loss_i.detach().unsqueeze(1), entb, rep_norm], dim=1)
                weights = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
                loss = (weights * loss_i).sum()
                optimizer_main.zero_grad()
                loss.backward()
                optimizer_main.step()
                # meta-update
                if step % N_meta == 0:
                    main_model.eval()
                    with torch.no_grad():
                        base_loss = ce_full(main_model(X_test), y_test).item()
                    feats_list, contr_list = [], []
                    base_state = main_model.state_dict()
                    # choose indices
                    if ablation == "random_sampling":
                        idxs = random.sample(range(len(X_train)), K_meta)
                    else:  # hardness
                        with torch.no_grad():
                            logits_all = main_model(X_train.to(device))
                            loss_all = (
                                crit_main(logits_all, y_train.to(device)).cpu().numpy()
                            )
                        idxs = loss_all.argsort()[::-1][:K_meta].tolist()
                    for idx in idxs:
                        xi = X_train[idx].unsqueeze(0).to(device)
                        yi = y_train[idx].unsqueeze(0).to(device)
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
                            new_loss = ce_full(clone(X_test), y_test).item()
                        contr_list.append([base_loss - new_loss])
                    feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(
                        device
                    )
                    contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(
                        device
                    )
                    for _ in range(5):
                        dvn_model.train()
                        p = dvn_model(feats_meta)
                        dvn_loss = crit_dvn(p, contr_meta)
                        optimizer_dvn.zero_grad()
                        dvn_loss.backward()
                        optimizer_dvn.step()
                    dvn_model.eval()
                    with torch.no_grad():
                        preds_meta = dvn_model(feats_meta).cpu().numpy().flatten()
                        true_meta = contr_meta.cpu().numpy().flatten()
                    corr = spearmanr(preds_meta, true_meta).correlation
                    ds_data["corrs"].append(corr)
                    if prev_corr is not None:
                        N_meta = (
                            min(50, N_meta * 2)
                            if corr > prev_corr
                            else max(1, N_meta // 2)
                        )
                    ds_data["N_meta_history"].append(N_meta)
                    prev_corr = corr
                    main_model.train()
                step += 1
            # end of epoch: evaluate train & val
            main_model.eval()
            # train metrics
            t_loss, t_corr, t_n = 0.0, 0, 0
            with torch.no_grad():
                for Xb, entb, yb in train_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    lo = ce_full(main_model(Xb), yb).item()
                    t_loss += lo * Xb.size(0)
                    preds_t = main_model(Xb).argmax(1)
                    t_corr += (preds_t == yb).sum().item()
                    t_n += Xb.size(0)
            train_loss = t_loss / t_n
            train_acc = t_corr / t_n
            ds_data["losses"]["train"].append(train_loss)
            ds_data["metrics"]["train"].append(train_acc)
            # val
            with torch.no_grad():
                logits_val = main_model(X_test)
                val_loss = ce_full(logits_val, y_test).item()
                val_acc = (logits_val.argmax(1) == y_test).float().mean().item()
            ds_data["losses"]["val"].append(val_loss)
            ds_data["metrics"]["val"].append(val_acc)
            ds_data["predictions"].append(logits_val.argmax(1).cpu().numpy())
            print(
                f"[{ablation}][{name}] Epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                f" val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
        experiment_data[ablation][name] = ds_data

# save all
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
