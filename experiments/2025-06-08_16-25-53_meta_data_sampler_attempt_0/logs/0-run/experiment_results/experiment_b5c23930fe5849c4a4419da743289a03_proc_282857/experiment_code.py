import os, random
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset registry
hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}


# architectures
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


class DVNLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


# prepare experiment_data structure
experiment_data = {"full_DVN": {}, "linear_DVN": {}}

# main loop over ablations
for ablation in experiment_data.keys():
    for ds_name, hf_name in hf_datasets.items():
        # load and preprocess
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

        # instantiate models
        main_model = MLP(input_dim, num_classes).to(device)
        dvn_model = (
            DVN().to(device) if ablation == "full_DVN" else DVNLinear().to(device)
        )
        opt_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
        opt_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
        crit_main = nn.CrossEntropyLoss(reduction="none")
        crit_dvn = nn.MSELoss()

        # history containers
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        corrs, N_meta_hist = [], []

        # meta‐training hyperparams
        N_meta, prev_corr = 10, None
        K_meta, epochs = 20, 3

        # training loop
        for epoch in range(epochs):
            main_model.train()
            step = 0
            for Xb, entb, yb in train_loader:
                Xb, entb, yb = Xb.to(device), entb.to(device), yb.to(device)
                logits = main_model(Xb)
                loss_i = crit_main(logits, yb)
                # rep_norm feature
                reps = main_model.net[1](main_model.net[0](Xb))
                rep_norm = torch.norm(reps, dim=1, keepdim=True)
                feats = torch.cat([loss_i.detach().unsqueeze(1), entb, rep_norm], dim=1)
                w = dvn_model(feats).squeeze(1)
                w = torch.softmax(w, dim=0)
                loss = (w * loss_i).sum()
                opt_main.zero_grad()
                loss.backward()
                opt_main.step()

                # meta update
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
                                clone(X_test), y_test_t
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
                        p = dvn_model(feats_meta)
                        l_dvn = crit_dvn(p, contr_meta)
                        opt_dvn.zero_grad()
                        l_dvn.backward()
                        opt_dvn.step()
                    dvn_model.eval()
                    with torch.no_grad():
                        preds = dvn_model(feats_meta).cpu().numpy().flatten()
                    true = contr_meta.cpu().numpy().flatten()
                    corr = spearmanr(preds, true).correlation
                    corrs.append(corr)
                    if prev_corr is not None:
                        N_meta = (
                            min(50, N_meta * 2)
                            if corr > prev_corr
                            else max(1, N_meta // 2)
                        )
                    N_meta_hist.append(N_meta)
                    prev_corr = corr
                    main_model.train()
                step += 1

            # end of epoch: eval train & val
            main_model.eval()
            with torch.no_grad():
                lt = nn.CrossEntropyLoss()(
                    main_model(X_train.to(device)), y_train_t.to(device)
                ).item()
                at = (
                    (main_model(X_train.to(device)).argmax(1) == y_train_t.to(device))
                    .float()
                    .mean()
                    .item()
                )
                lv = nn.CrossEntropyLoss()(main_model(X_test), y_test_t).item()
                av = (main_model(X_test).argmax(1) == y_test_t).float().mean().item()
            train_losses.append(lt)
            train_accs.append(at)
            val_losses.append(lv)
            val_accs.append(av)
            print(
                f"[{ablation}][{ds_name}] Epoch {epoch}: tr_loss={lt:.4f}, tr_acc={at:.4f}, val_loss={lv:.4f}, val_acc={av:.4f}"
            )

        # final predictions
        main_model.eval()
        with torch.no_grad():
            preds = main_model(X_test).argmax(1).cpu().numpy()
        gts = y_test_t.cpu().numpy()

        # store
        experiment_data[ablation][ds_name] = {
            "metrics": {"train": train_accs, "val": val_accs},
            "losses": {"train": train_losses, "val": val_losses},
            "predictions": preds,
            "ground_truth": gts,
            "corrs": corrs,
            "N_meta_history": N_meta_hist,
        }

# save all results
np.save("experiment_data.npy", experiment_data)
print("Saved experiment_data.npy")
