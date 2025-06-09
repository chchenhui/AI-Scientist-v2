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


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DVN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


for name, hf_name in hf_datasets.items():
    ds = load_dataset(hf_name)
    train = ds["train"].shuffle(seed=42).select(range(1000))
    test = ds["test"].shuffle(seed=42).select(range(200))
    text_col = "text" if "text" in train.column_names else "content"
    train_texts, test_texts = train[text_col], test[text_col]
    y_train, y_test = train["label"], test["label"]

    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(train_texts + test_texts)
    X_train_np = tfidf.transform(train_texts).toarray()
    X_test_np = tfidf.transform(test_texts).toarray()
    mean_vec = X_train_np.mean(axis=0, keepdims=True)
    div_train_np = np.linalg.norm(X_train_np - mean_vec, axis=1)

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    div_train_tensor = torch.tensor(div_train_np, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train_tensor, div_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_dim = X_train_np.shape[1]
    num_classes = len(set(y_train))
    main_model = MLP(input_dim, num_classes).to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
    criterion_main = nn.CrossEntropyLoss(reduction="none")
    criterion_main_mean = nn.CrossEntropyLoss(reduction="mean")
    criterion_dvn = nn.MSELoss()

    meta_freq, N_min, N_max = 10, 1, 50
    low_th, high_th = 0.2, 0.8
    K_meta, meta_steps, epochs = 20, 3, 3

    experiment_data[name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "corrs": [],
    }

    for epoch in range(epochs):
        main_model.train()
        train_loss_sum, batches = 0.0, 0
        step = 0
        for Xb, divb, yb in train_loader:
            Xb, divb, yb = Xb.to(device), divb.to(device), yb.to(device)
            logits = main_model(Xb)
            loss_i = criterion_main(logits, yb)
            feats_i = torch.cat([loss_i.detach().unsqueeze(1), divb], dim=1)
            scores = dvn_model(feats_i).squeeze(1)
            weights = torch.softmax(scores, dim=0)
            weighted_loss = (weights * loss_i).sum()
            optimizer_main.zero_grad()
            weighted_loss.backward()
            optimizer_main.step()
            train_loss_sum += weighted_loss.item()
            batches += 1

            if step % meta_freq == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = criterion_main_mean(main_model(X_test), y_test).item()
                feats_list, contr_list = [], []
                base_state = main_model.state_dict()
                for idx in random.sample(range(len(X_train_tensor)), K_meta):
                    xi = X_train_tensor[idx].unsqueeze(0).to(device)
                    yi = y_train_tensor[idx].unsqueeze(0).to(device)
                    with torch.no_grad():
                        li = criterion_main(main_model(xi), yi).item()
                    div_i = div_train_tensor[idx].item()
                    feats_list.append([li, div_i])
                    clone = MLP(input_dim, num_classes).to(device)
                    clone.load_state_dict(base_state)
                    opt_c = torch.optim.Adam(clone.parameters(), lr=1e-3)
                    clone.train()
                    out = clone(xi)
                    lci = criterion_main(out, yi).mean()
                    opt_c.zero_grad()
                    lci.backward()
                    opt_c.step()
                    clone.eval()
                    with torch.no_grad():
                        new_loss = criterion_main_mean(clone(X_test), y_test).item()
                    contr_list.append([base_loss - new_loss])
                feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(device)
                contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(device)
                for _ in range(meta_steps):
                    dvn_model.train()
                    pred = dvn_model(feats_meta)
                    dvn_loss = criterion_dvn(pred, contr_meta)
                    optimizer_dvn.zero_grad()
                    dvn_loss.backward()
                    optimizer_dvn.step()
                dvn_model.eval()
                with torch.no_grad():
                    preds_c = dvn_model(feats_meta).cpu().numpy().flatten()
                true_c = contr_meta.cpu().numpy().flatten()
                corr = spearmanr(preds_c, true_c).correlation
                experiment_data[name]["predictions"].extend(preds_c.tolist())
                experiment_data[name]["ground_truth"].extend(true_c.tolist())
                experiment_data[name]["corrs"].append(corr)
                if corr < low_th:
                    meta_freq = max(N_min, meta_freq // 2)
                elif corr > high_th:
                    meta_freq = min(N_max, meta_freq * 2)
                print(f"[{name}] Step {step}: Spearman Corr={corr:.4f}")
                main_model.train()
            step += 1

        main_model.eval()
        with torch.no_grad():
            tr_logits = main_model(X_train_tensor.to(device))
            tr_loss = criterion_main_mean(tr_logits, y_train_tensor.to(device)).item()
            tr_acc = (
                (tr_logits.argmax(dim=1) == y_train_tensor.to(device))
                .float()
                .mean()
                .item()
            )
            val_logits = main_model(X_test)
            val_loss = criterion_main_mean(val_logits, y_test).item()
            val_acc = (val_logits.argmax(dim=1) == y_test).float().mean().item()
        experiment_data[name]["losses"]["train"].append(tr_loss)
        experiment_data[name]["metrics"]["train"].append(tr_acc)
        experiment_data[name]["losses"]["val"].append(val_loss)
        experiment_data[name]["metrics"]["val"].append(val_acc)
        print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
