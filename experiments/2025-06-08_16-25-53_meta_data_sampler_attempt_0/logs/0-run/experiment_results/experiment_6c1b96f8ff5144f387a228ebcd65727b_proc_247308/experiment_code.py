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

hf_datasets = {"ag_news": "ag_news", "dbpedia": "dbpedia_14", "yelp": "yelp_polarity"}
experiment_data = {}


class MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp, 128), nn.ReLU(), nn.Linear(128, out))

    def forward(self, x):
        return self.net(x)


class DVN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


for name, hf in hf_datasets.items():
    ds = load_dataset(hf)
    tr = ds["train"].shuffle(seed=42).select(range(1000))
    te = ds["test"].shuffle(seed=42).select(range(200))
    text_col = "text" if "text" in tr.column_names else "content"
    tv = TfidfVectorizer(max_features=500, norm="l2")
    tv.fit(tr[text_col] + te[text_col])
    Xtr = tv.transform(tr[text_col]).toarray()
    Xte = tv.transform(te[text_col]).toarray()
    mean_vec = Xtr.mean(0, keepdims=True)
    div_tr = np.linalg.norm(Xtr - mean_vec, axis=1)
    X_train = torch.tensor(Xtr, dtype=torch.float32)
    div_train = torch.tensor(div_tr, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(tr["label"], dtype=torch.long)
    X_test = torch.tensor(Xte, dtype=torch.float32).to(device)
    y_test = torch.tensor(te["label"], dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train, div_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_dim = X_train.shape[1]
    num_classes = len(set(tr["label"]))
    main_model = MLP(input_dim, num_classes).to(device)
    dvn_model = DVN().to(device)
    opt_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    opt_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
    crit_main = nn.CrossEntropyLoss(reduction="none")
    crit_main_mean = nn.CrossEntropyLoss(reduction="mean")
    crit_dvn = nn.MSELoss()

    meta_freq, N_min, N_max = 50, 10, 200
    low_th, high_th = 0.3, 0.7
    K_meta, meta_steps, epochs = 30, 5, 3

    experiment_data[name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "corrs": [],
    }

    for epoch in range(epochs):
        main_model.train()
        step = 0
        for Xb, divb, yb in train_loader:
            Xb, divb, yb = Xb.to(device), divb.to(device), yb.to(device)
            Xb.requires_grad_(True)
            logits = main_model(Xb)
            loss_i = crit_main(logits, yb)
            grads = torch.autograd.grad(loss_i.sum(), Xb, retain_graph=True)[0]
            grad_norm = grads.norm(dim=1).detach().unsqueeze(1)
            feats = torch.cat([loss_i.detach().unsqueeze(1), grad_norm, divb], dim=1)
            scores = dvn_model(feats).squeeze(1)
            weights = torch.softmax(scores, dim=0)
            wloss = (weights * loss_i).sum()
            opt_main.zero_grad()
            wloss.backward()
            opt_main.step()

            if step % meta_freq == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = crit_main_mean(main_model(X_test), y_test).item()
                feats_list, contr_list = [], []
                base_state = main_model.state_dict()
                idxs = random.sample(range(len(X_train)), K_meta)
                for idx in idxs:
                    # clone sample and enable gradient tracking
                    xi = (
                        X_train[idx : idx + 1]
                        .clone()
                        .detach()
                        .to(device)
                        .requires_grad_(True)
                    )
                    yi = y_train[idx : idx + 1].to(device)
                    li = crit_main(main_model(xi), yi).item()
                    gi = (
                        torch.autograd.grad(crit_main(main_model(xi), yi).sum(), xi)[0]
                        .norm()
                        .item()
                    )
                    di = div_train[idx].item()
                    feats_list.append([li, gi, di])

                    # compute one-step influence by cloning model
                    clone = MLP(input_dim, num_classes).to(device)
                    clone.load_state_dict(base_state)
                    oc = torch.optim.Adam(clone.parameters(), lr=1e-3)
                    oc.zero_grad()
                    lci = crit_main(clone(xi), yi).sum()
                    lci.backward()
                    oc.step()
                    clone.eval()
                    with torch.no_grad():
                        new_loss = crit_main_mean(clone(X_test), y_test).item()
                    contr_list.append(base_loss - new_loss)

                feats_m = torch.tensor(feats_list, dtype=torch.float32).to(device)
                true_c = (
                    torch.tensor(contr_list, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                )
                for _ in range(meta_steps):
                    dvn_model.train()
                    pred_c = dvn_model(feats_m)
                    loss_d = crit_dvn(pred_c, true_c)
                    opt_dvn.zero_grad()
                    loss_d.backward()
                    opt_dvn.step()
                dvn_model.eval()
                with torch.no_grad():
                    pc = dvn_model(feats_m).cpu().numpy().flatten()
                tc = true_c.cpu().numpy().flatten()
                corr = spearmanr(pc, tc).correlation
                experiment_data[name]["corrs"].append(corr)
                print(f"[{name}] Step {step}: Spearman Corr={corr:.4f}")
                if corr < low_th:
                    meta_freq = max(N_min, meta_freq // 2)
                elif corr > high_th:
                    meta_freq = min(N_max, meta_freq * 2)
                main_model.train()
            step += 1

        main_model.eval()
        with torch.no_grad():
            tr_l = crit_main_mean(
                main_model(X_train.to(device)), y_train.to(device)
            ).item()
            tr_acc = (
                (main_model(X_train.to(device)).argmax(1) == y_train.to(device))
                .float()
                .mean()
                .item()
            )
            va_l = crit_main_mean(main_model(X_test), y_test).item()
            va_acc = (main_model(X_test).argmax(1) == y_test).float().mean().item()
        experiment_data[name]["losses"]["train"].append(tr_l)
        experiment_data[name]["metrics"]["train"].append(tr_acc)
        experiment_data[name]["losses"]["val"].append(va_l)
        experiment_data[name]["metrics"]["val"].append(va_acc)
        print(f"Epoch {epoch}: validation_loss = {va_l:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
