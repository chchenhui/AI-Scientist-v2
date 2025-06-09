import os, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import spearmanr

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# datasets and noise levels
hf_datasets = {"ag_news": "ag_news", "yelp": "yelp_polarity", "dbpedia": "dbpedia_14"}
noise_levels = [0.1, 0.2, 0.5]
experiment_data = {"Ablate_Label_Noise_Robustness": {}}

for name, hf_name in hf_datasets.items():
    # load and preprocess
    ds = load_dataset(hf_name)
    train = ds["train"].shuffle(seed=seed).select(range(1000))
    test = ds["test"].shuffle(seed=seed).select(range(200))
    text_col = "text" if "text" in train.column_names else "content"
    tr_txt, te_txt = train[text_col], test[text_col]
    y_tr, y_te = train["label"], test["label"]
    tfidf = TfidfVectorizer(max_features=500, norm="l2")
    tfidf.fit(tr_txt + te_txt)
    X_tr_np = tfidf.transform(tr_txt).toarray()
    X_te_np = tfidf.transform(te_txt).toarray()
    ent_tr = -np.sum(X_tr_np * np.log(X_tr_np + 1e-10), axis=1, keepdims=True)
    # constants
    input_dim = X_tr_np.shape[1]
    num_classes = len(set(y_tr))

    # model defs
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

    # per‐noise experiments
    dataset_results = {}
    for noise in noise_levels:
        print(f"Running {name} with {int(noise*100)}% label noise")
        # inject noise
        y_tr_np = np.array(y_tr, copy=True)
        n_flip = int(len(y_tr_np) * noise)
        flip_idx = np.random.choice(len(y_tr_np), n_flip, replace=False)
        for i in flip_idx:
            orig = y_tr_np[i]
            choices = list(range(num_classes))
            choices.remove(orig)
            y_tr_np[i] = np.random.choice(choices)
        # tensors
        X_train = torch.tensor(X_tr_np, dtype=torch.float32)
        ent_train = torch.tensor(ent_tr, dtype=torch.float32)
        y_train_t = torch.tensor(y_tr_np, dtype=torch.long)
        X_test = torch.tensor(X_te_np, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_te, dtype=torch.long).to(device)
        # loaders
        train_ds = TensorDataset(X_train, ent_train, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        # instantiate
        main_model = MLP().to(device)
        dvn_model = DVN().to(device)
        optim_main = torch.optim.Adam(main_model.parameters(), lr=1e-3)
        optim_dvn = torch.optim.Adam(dvn_model.parameters(), lr=1e-3)
        crit_main = nn.CrossEntropyLoss(reduction="none")
        crit_dvn = nn.MSELoss()
        # logging containers
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        corrs, N_meta_hist = [], []
        # meta parameters
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
                weights = torch.softmax(dvn_model(feats).squeeze(1), dim=0)
                loss = (weights * loss_i).sum()
                optim_main.zero_grad()
                loss.backward()
                optim_main.step()

                # meta‐update
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
                            rn = torch.norm(rep_i, dim=1).item()
                        feats_list.append([li, ent_i, rn])
                        clone = MLP().to(device)
                        clone.load_state_dict(base_state)
                        opt_c = torch.optim.Adam(clone.parameters(), lr=1e-3)
                        clone.train()
                        lc = nn.CrossEntropyLoss()(clone(xi), yi)
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
                        l_d = crit_dvn(p, contr_meta)
                        optim_dvn.zero_grad()
                        l_d.backward()
                        optim_dvn.step()
                    dvn_model.eval()
                    with torch.no_grad():
                        preds_m = dvn_model(feats_meta).cpu().numpy().flatten()
                    true_m = contr_meta.cpu().numpy().flatten()
                    corr = spearmanr(preds_m, true_m).correlation
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

            # record train metrics
            main_model.eval()
            total_loss = total_correct = total_samples = 0
            eval_loss_fn = nn.CrossEntropyLoss(reduction="sum")
            with torch.no_grad():
                for Xb, _, yb in train_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    out = main_model(Xb)
                    total_loss += eval_loss_fn(out, yb).item()
                    total_correct += (out.argmax(1) == yb).sum().item()
                    total_samples += yb.size(0)
                train_losses.append(total_loss / total_samples)
                train_accs.append(total_correct / total_samples)
                # val metrics
                out_val = main_model(X_test)
                val_losses.append(nn.CrossEntropyLoss()(out_val, y_test_t).item())
                val_accs.append((out_val.argmax(1) == y_test_t).float().mean().item())

            print(
                f"[{name} noise={int(noise*100)}%] Epoch {epoch}: "
                f"train_loss={train_losses[-1]:.4f}, train_acc={train_accs[-1]:.4f}, "
                f"val_loss={val_losses[-1]:.4f}, val_acc={val_accs[-1]:.4f}"
            )

        # final test predictions
        main_model.eval()
        with torch.no_grad():
            logits_test = main_model(X_test)
            preds = logits_test.argmax(1).cpu().numpy()
            gt = y_test_t.cpu().numpy()

        # save results
        dataset_results[str(int(noise * 100))] = {
            "metrics": {"train": np.array(train_accs), "val": np.array(val_accs)},
            "losses": {"train": np.array(train_losses), "val": np.array(val_losses)},
            "corrs": np.array(corrs),
            "N_meta_history": np.array(N_meta_hist),
            "predictions": preds,
            "ground_truth": gt,
        }
    experiment_data["Ablate_Label_Noise_Robustness"][name] = dataset_results

# save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
