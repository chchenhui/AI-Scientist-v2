import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr
import random

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Datasets to test
hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}
experiment_data = {}

for name, hf_name in hf_datasets.items():
    ds = load_dataset(hf_name)
    train = ds["train"].shuffle(seed=42).select(range(1000))
    test = ds["test"].shuffle(seed=42).select(range(200))
    text_col = "text" if "text" in train.column_names else "content"
    train_texts, test_texts = train[text_col], test[text_col]
    y_train, y_test = train["label"], test["label"]

    # TF-IDF features and entropy
    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(train_texts + test_texts)
    X_train_np = tfidf.transform(train_texts).toarray()
    X_test_np = tfidf.transform(test_texts).toarray()
    ent_train_np = -np.sum(X_train_np * np.log(X_train_np + 1e-10), axis=1)

    # Tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    ent_train = torch.tensor(ent_train_np, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train, ent_train, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    input_dim = X_train.shape[1]
    num_classes = len(set(y_train))

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
            self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x)

    main_model = MLP().to(device)
    dvn_model = DVN().to(device)
    optimizer_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
    optimizer_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
    criterion_main = nn.CrossEntropyLoss(reduction="none")
    criterion_dvn = nn.MSELoss()

    experiment_data[name] = {"val_loss": [], "val_acc": [], "corrs": []}

    N_meta, K_meta, epochs = 10, 20, 3
    for epoch in range(epochs):
        main_model.train()
        step = 0
        for Xb, entb, yb in train_loader:
            Xb, entb, yb = Xb.to(device), entb.to(device), yb.to(device)
            logits = main_model(Xb)
            loss_i = criterion_main(logits, yb)
            feats = torch.cat([loss_i.detach().unsqueeze(1), entb], dim=1)
            w = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
            loss = (w * loss_i).sum()
            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()

            if step % N_meta == 0:
                main_model.eval()
                with torch.no_grad():
                    base_loss = nn.CrossEntropyLoss(reduction="mean")(
                        main_model(X_test), y_test_t
                    ).item()
                feats_list, contr_list = [], []
                base_state = main_model.state_dict()
                for idx in random.sample(range(len(X_train)), K_meta):
                    xi = X_train[idx].unsqueeze(0).to(device)
                    yi = torch.tensor([y_train_t[idx]], dtype=torch.long).to(device)
                    with torch.no_grad():
                        li = criterion_main(main_model(xi), yi).item()
                    ent_i = ent_train[idx].item()
                    feats_list.append([li, ent_i])

                    clone = MLP().to(device)
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
                        new_loss = nn.CrossEntropyLoss(reduction="mean")(
                            clone(X_test), y_test_t
                        ).item()
                    contr_list.append([base_loss - new_loss])

                feats_meta = torch.tensor(feats_list, dtype=torch.float32).to(device)
                contr_meta = torch.tensor(contr_list, dtype=torch.float32).to(device)
                for _ in range(5):
                    dvn_model.train()
                    p = dvn_model(feats_meta)
                    dvn_loss = criterion_dvn(p, contr_meta)
                    optimizer_dvn.zero_grad()
                    dvn_loss.backward()
                    optimizer_dvn.step()
                dvn_model.eval()
                with torch.no_grad():
                    preds_c = dvn_model(feats_meta).cpu().numpy().flatten()
                true_c = contr_meta.cpu().numpy().flatten()
                corr = spearmanr(preds_c, true_c).correlation
                experiment_data[name]["corrs"].append(corr)
                print(f"[{name}] Step {step}: Spearman Corr={corr:.4f}")
                main_model.train()
            step += 1

        main_model.eval()
        with torch.no_grad():
            logits_val = main_model(X_test)
            val_loss = nn.CrossEntropyLoss(reduction="mean")(
                logits_val, y_test_t
            ).item()
            acc = (logits_val.argmax(dim=1) == y_test_t).float().mean().item()
        experiment_data[name]["val_loss"].append(val_loss)
        experiment_data[name]["val_acc"].append(acc)
        print(f"[{name}] Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={acc:.4f}")

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
