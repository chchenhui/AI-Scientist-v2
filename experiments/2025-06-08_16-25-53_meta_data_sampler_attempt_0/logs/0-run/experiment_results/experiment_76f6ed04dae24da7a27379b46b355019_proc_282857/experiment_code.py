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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}

# prepare experiment data container
abl_name = "Ablate_Shared_TestSet_For_Meta"
experiment_data = {abl_name: {}}

for name, hf_name in hf_datasets.items():
    # load and subsample
    ds = load_dataset(hf_name)
    train = ds["train"].shuffle(seed=42).select(range(1000))
    test_all = ds["test"].shuffle(seed=42).select(range(200))
    # split test into meta‐validation and final evaluation
    n_meta = len(test_all) // 2
    meta = test_all.select(range(n_meta))
    final = test_all.select(range(n_meta, len(test_all)))
    text_col = "text" if "text" in train.column_names else "content"
    tr_txt, me_txt, fe_txt = train[text_col], meta[text_col], final[text_col]
    y_tr, y_me, y_fe = train["label"], meta["label"], final["label"]

    # vectorize
    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(tr_txt + me_txt + fe_txt)
    X_tr_np = tfidf.transform(tr_txt).toarray()
    X_me_np = tfidf.transform(me_txt).toarray()
    X_fe_np = tfidf.transform(fe_txt).toarray()
    ent_tr = -np.sum(X_tr_np * np.log(X_tr_np + 1e-10), axis=1, keepdims=True)

    # tensors
    X_train = torch.tensor(X_tr_np, dtype=torch.float32)
    ent_train = torch.tensor(ent_tr, dtype=torch.float32)
    y_train = torch.tensor(y_tr, dtype=torch.long)
    X_meta = torch.tensor(X_me_np, dtype=torch.float32).to(device)
    y_meta = torch.tensor(y_me, dtype=torch.long).to(device)
    X_eval = torch.tensor(X_fe_np, dtype=torch.float32).to(device)
    y_eval = torch.tensor(y_fe, dtype=torch.long).to(device)

    # dataloader
    train_ds = TensorDataset(X_train, ent_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # models
    input_dim = X_train.shape[1]
    num_classes = len(set(y_tr))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    class DVN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x)

    main_model = MLP().to(device)
    dvn = DVN().to(device)
    opt_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    opt_dvn = torch.optim.Adam(dvn.parameters(), lr=1e-3)
    crit_main = nn.CrossEntropyLoss(reduction="none")
    crit_dvn = nn.MSELoss()

    # logging
    experiment_data[abl_name][name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "corrs": [],
        "N_meta_history": [],
        "predictions": [],
        "ground_truth": [],
    }
    ab_data = experiment_data[abl_name][name]

    # meta‐learning hyperparams
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
            reps = main_model.net[1](main_model.net[0](Xb))
            rep_norm = torch.norm(reps, dim=1, keepdim=True)
            feats = torch.cat([loss_i.detach().unsqueeze(1), entb, rep_norm], dim=1)
            weights = torch.softmax(dvn(feats).squeeze(1), dim=0)
            loss = (weights * loss_i).sum()
            opt_main.zero_grad()
            loss.backward()
            opt_main.step()

            # meta update
            if step % N_meta == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = nn.CrossEntropyLoss()(main_model(X_meta), y_meta).item()
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
                        new_l = nn.CrossEntropyLoss()(clone(X_meta), y_meta).item()
                    contr_list.append([base_loss - new_l])
                feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(device)
                contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(device)
                for _ in range(5):
                    dvn.train()
                    dvn_loss = crit_dvn(dvn(feats_meta), contr_meta)
                    opt_dvn.zero_grad()
                    dvn_loss.backward()
                    opt_dvn.step()
                dvn.eval()
                with torch.no_grad():
                    preds = dvn(feats_meta).cpu().numpy().flatten()
                true = contr_meta.cpu().numpy().flatten()
                corr = spearmanr(preds, true).correlation
                ab_data["corrs"].append(corr)
                if prev_corr is not None:
                    N_meta = (
                        min(50, N_meta * 2) if corr > prev_corr else max(1, N_meta // 2)
                    )
                ab_data["N_meta_history"].append(N_meta)
                prev_corr = corr
                main_model.train()
            step += 1

        # evaluate train set metrics
        main_model.eval()
        with torch.no_grad():
            logits_tr = main_model(X_train.to(device))
            l_tr = nn.CrossEntropyLoss()(logits_tr, y_train.to(device)).item()
            a_tr = (logits_tr.argmax(1) == y_train.to(device)).float().mean().item()
        ab_data["losses"]["train"].append(l_tr)
        ab_data["metrics"]["train"].append(a_tr)

        # final eval metrics
        with torch.no_grad():
            logits_val = main_model(X_eval)
            l_val = nn.CrossEntropyLoss()(logits_val, y_eval).item()
            a_val = (logits_val.argmax(1) == y_eval).float().mean().item()
        ab_data["losses"]["val"].append(l_val)
        ab_data["metrics"]["val"].append(a_val)
        print(
            f"[{name}] Epoch {epoch}: train_loss={l_tr:.4f}, train_acc={a_tr:.4f}, val_loss={l_val:.4f}, val_acc={a_val:.4f}"
        )

    # final predictions
    main_model.eval()
    with torch.no_grad():
        final_logits = main_model(X_eval)
        preds = final_logits.argmax(1).cpu().numpy()
    ab_data["predictions"].append(preds)
    ab_data["ground_truth"].append(y_fe)

# save all data
np.save("experiment_data.npy", experiment_data, allow_pickle=True)
print("Saved experiment_data.npy")
