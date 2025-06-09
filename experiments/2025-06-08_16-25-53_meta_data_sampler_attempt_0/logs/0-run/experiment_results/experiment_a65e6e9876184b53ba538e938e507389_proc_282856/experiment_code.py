import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# reproducibility & device
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# experiment storage
experiment_data = {"Ablate_Meta_Inner_Update_Steps": {}}

hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}

for name, hf_name in hf_datasets.items():
    ds = load_dataset(hf_name)
    train = ds["train"].shuffle(seed=42).select(range(1000))
    test = ds["test"].shuffle(seed=42).select(range(200))
    text_col = "text" if "text" in train.column_names else "content"
    tr_txt, te_txt = train[text_col], test[text_col]
    y_tr, y_te = train["label"], test["label"]

    # TF-IDF + entropy
    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(tr_txt + te_txt)
    X_tr = torch.tensor(tfidf.transform(tr_txt).toarray(), dtype=torch.float32)
    ent_tr = torch.tensor(
        -np.sum(
            tfidf.transform(tr_txt).toarray()
            * np.log(tfidf.transform(tr_txt).toarray() + 1e-10),
            axis=1,
            keepdims=True,
        ),
        dtype=torch.float32,
    )
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te = torch.tensor(tfidf.transform(te_txt).toarray(), dtype=torch.float32).to(
        device
    )
    y_te_t = torch.tensor(y_te, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_tr, ent_tr, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_dim = X_tr.shape[1]
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
    dvn_model = DVN().to(device)
    opt_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    opt_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
    crit_ce = nn.CrossEntropyLoss(reduction="none")
    crit_mse = nn.MSELoss()

    exp = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": y_te_t.cpu().numpy(),
        "corrs": [],
        "N_meta_history": [],
    }
    experiment_data["Ablate_Meta_Inner_Update_Steps"][name] = exp

    N_meta, prev_corr = 10, None
    K_meta, epochs = 20, 3
    inner_lr = 1e-2

    for epoch in range(epochs):
        main_model.train()
        step = 0
        for Xb, entb, yb in train_loader:
            Xb, entb, yb = Xb.to(device), entb.to(device), yb.to(device)
            logits = main_model(Xb)
            loss_i = crit_ce(logits, yb)
            reps = main_model.net[1](main_model.net[0](Xb))
            rep_norm = torch.norm(reps, dim=1, keepdim=True)
            feats = torch.cat([loss_i.detach().unsqueeze(1), entb, rep_norm], dim=1)
            weights = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
            loss = (weights * loss_i).sum()
            opt_main.zero_grad()
            loss.backward()
            opt_main.step()

            if step % N_meta == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = nn.CrossEntropyLoss()(main_model(X_te), y_te_t).item()
                feats_list, contr_list = [], []
                base_state = main_model.state_dict()
                for idx in random.sample(range(len(X_tr)), K_meta):
                    xi = X_tr[idx].unsqueeze(0).to(device)
                    yi = y_tr_t[idx].unsqueeze(0).to(device)
                    with torch.no_grad():
                        li = crit_ce(main_model(xi), yi).item()
                        ent_i = ent_tr[idx].item()
                        rep_i = main_model.net[1](main_model.net[0](xi))
                        rep_norm_i = torch.norm(rep_i, dim=1).item()
                    feats_list.append([li, ent_i, rep_norm_i])

                    # clone + INNER update
                    clone = MLP().to(device)
                    clone.load_state_dict(base_state)
                    clone_opt = torch.optim.SGD(clone.parameters(), lr=inner_lr)
                    clone.train()
                    out = clone(xi)
                    inner_loss = nn.CrossEntropyLoss()(out, yi)
                    clone_opt.zero_grad()
                    inner_loss.backward()
                    clone_opt.step()
                    clone.eval()

                    with torch.no_grad():
                        new_loss = nn.CrossEntropyLoss()(clone(X_te), y_te_t).item()
                    contr_list.append([base_loss - new_loss])

                feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(device)
                contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(device)
                for _ in range(5):
                    dvn_model.train()
                    l_dvn = crit_mse(dvn_model(feats_meta), contr_meta)
                    opt_dvn.zero_grad()
                    l_dvn.backward()
                    opt_dvn.step()
                dvn_model.eval()
                with torch.no_grad():
                    preds = dvn_model(feats_meta).cpu().numpy().flatten()
                corr = spearmanr(preds, contr_meta.cpu().numpy().flatten()).correlation
                exp["corrs"].append(corr)
                if prev_corr is not None and corr is not None:
                    N_meta = (
                        min(50, N_meta * 2) if corr > prev_corr else max(1, N_meta // 2)
                    )
                exp["N_meta_history"].append(N_meta)
                prev_corr = corr
                print(
                    f"[{name}] Step {step}: Spearman Corr={corr:.4f}, N_meta={N_meta}"
                )
                main_model.train()
            step += 1

        # epoch eval
        main_model.eval()
        with torch.no_grad():
            lt = nn.CrossEntropyLoss()(
                main_model(X_tr.to(device)), y_tr_t.to(device)
            ).item()
            at = (
                (main_model(X_tr.to(device)).argmax(1) == y_tr_t.to(device))
                .float()
                .mean()
                .item()
            )
            lv = nn.CrossEntropyLoss()(main_model(X_te), y_te_t).item()
            av = (main_model(X_te).argmax(1) == y_te_t).float().mean().item()
        exp["metrics"]["train"].append(at)
        exp["metrics"]["val"].append(av)
        exp["losses"]["train"].append(lt)
        exp["losses"]["val"].append(lv)
        exp["predictions"].append(main_model(X_te).argmax(1).cpu().numpy())
        print(
            f"[{name}] Epoch {epoch}: train_loss={lt:.4f}, train_acc={at:.4f}, val_loss={lv:.4f}, val_acc={av:.4f}"
        )

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
