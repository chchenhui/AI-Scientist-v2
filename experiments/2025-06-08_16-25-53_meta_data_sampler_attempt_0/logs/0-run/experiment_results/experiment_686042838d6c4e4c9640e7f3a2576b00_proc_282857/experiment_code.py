import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

# prepare working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# define datasets and ablation variants
hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}
ablation_types = ["baseline", "Ablate_Weight_Softmax_Normalization"]

# initialize experiment data structure
experiment_data = {
    abl: {
        ds: {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
            "corrs": [],
            "N_meta_history": [],
        }
        for ds in hf_datasets
    }
    for abl in ablation_types
}


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


# run experiments
for abl in ablation_types:
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

        # init models, optimizers, losses
        main_model = MLP(input_dim, num_classes).to(device)
        dvn_model = DVN().to(device)
        opt_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
        opt_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
        crit_main = nn.CrossEntropyLoss(reduction="none")
        crit_dvn = nn.MSELoss()

        # meta params
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
                # representation norm feature
                rep = main_model.net[1](main_model.net[0](Xb))
                rep_norm = torch.norm(rep, dim=1, keepdim=True)
                feats = torch.cat([loss_i.detach().unsqueeze(1), entb, rep_norm], dim=1)
                raw = dvn_model(feats).squeeze(1)
                if abl == "baseline":
                    weights = torch.softmax(raw, dim=0)
                else:
                    weights = torch.sigmoid(raw)
                loss = (weights * loss_i).sum()
                opt_main.zero_grad()
                loss.backward()
                opt_main.step()

                # meta-update
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
                        dvn_loss = crit_dvn(p, contr_meta)
                        opt_dvn.zero_grad()
                        dvn_loss.backward()
                        opt_dvn.step()
                    dvn_model.eval()
                    with torch.no_grad():
                        preds = dvn_model(feats_meta).cpu().numpy().flatten()
                    true = contr_meta.cpu().numpy().flatten()
                    corr = spearmanr(preds, true).correlation
                    experiment_data[abl][name]["corrs"].append(corr)
                    if prev_corr is not None:
                        N_meta = (
                            min(50, N_meta * 2)
                            if corr > prev_corr
                            else max(1, N_meta // 2)
                        )
                    experiment_data[abl][name]["N_meta_history"].append(N_meta)
                    prev_corr = corr
                    print(
                        f"[{abl}][{name}] Step {step}: Corr={corr:.4f}, N_meta={N_meta}"
                    )
                    main_model.train()
                step += 1

            # end of epoch: eval
            main_model.eval()
            with torch.no_grad():
                # train metrics
                X_tr_dev = X_train.to(device)
                y_tr_dev = y_train_t.to(device)
                out_tr = main_model(X_tr_dev)
                tr_loss = nn.CrossEntropyLoss()(out_tr, y_tr_dev).item()
                tr_acc = (out_tr.argmax(1) == y_tr_dev).float().mean().item()
                # val metrics
                out_val = main_model(X_test)
                val_loss = nn.CrossEntropyLoss()(out_val, y_test_t).item()
                val_acc = (out_val.argmax(1) == y_test_t).float().mean().item()
            experiment_data[abl][name]["losses"]["train"].append(tr_loss)
            experiment_data[abl][name]["losses"]["val"].append(val_loss)
            experiment_data[abl][name]["metrics"]["train"].append(tr_acc)
            experiment_data[abl][name]["metrics"]["val"].append(val_acc)
            print(
                f"[{abl}][{name}] Epoch {epoch}: tr_loss={tr_loss:.4f}, tr_acc={tr_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # final preds & ground truth
        main_model.eval()
        with torch.no_grad():
            final_preds = main_model(X_test).argmax(1).cpu().numpy()
        experiment_data[abl][name]["predictions"] = final_preds
        experiment_data[abl][name]["ground_truth"] = y_test_t.cpu().numpy()

# save all
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
