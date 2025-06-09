import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# reproducibility
np.random.seed(0)
torch.manual_seed(0)

# synthetic data
N, D = 2000, 2
X = np.random.randn(N, D)
w_true, b_true = np.array([2.0, -3.0]), 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (np.random.rand(N) < probs).astype(int)

# split
idx = np.random.permutation(N)
tr_idx, va_idx, te_idx = idx[:1200], idx[1200:1500], idx[1500:]
X_train, y_train = X[tr_idx], y[tr_idx]
X_val, y_val = X[va_idx], y[va_idx]
X_test, y_test = X[te_idx], y[te_idx]

# normalize
m, s = X_train.mean(0), X_train.std(0) + 1e-6
X_train = (X_train - m) / s
X_val = (X_val - m) / s
X_test = (X_test - m) / s


# datasets
class SimpleDS(Dataset):
    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class UserDS(Dataset):
    def __init__(self, X, y):
        self.feat = torch.from_numpy(X).float()
        self.label = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return {"feat": self.feat[i], "label": self.label[i]}


class SoftLabelDS(Dataset):
    def __init__(self, X, soft, hard):
        self.feat = torch.from_numpy(X).float()
        self.soft = torch.from_numpy(soft).float()
        self.hard = torch.from_numpy(hard).long()

    def __len__(self):
        return len(self.hard)

    def __getitem__(self, i):
        return {"feat": self.feat[i], "soft": self.soft[i], "hard": self.hard[i]}


# models
class AIModel(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


class UserModel(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


# Jensen-Shannon divergence
def js_div(p, q):
    m = 0.5 * (p + q)
    kl1 = np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12)), axis=1)
    kl2 = np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)), axis=1)
    return 0.5 * (kl1 + kl2)


# experiment storage
experiment_data = {
    "CE_hard_labels": {},
    "soft_label_distillation": {},
    "bias_awareness": {},
    "dual_channel": {},
}

ai_batch_sizes = [16, 32, 64]
usr_batch_sizes = [16, 32, 64]

for ai_bs in ai_batch_sizes:
    # train AI
    ai_tr = DataLoader(SimpleDS(X_train, y_train), batch_size=ai_bs, shuffle=True)
    ai_val = DataLoader(SimpleDS(X_val, y_val), batch_size=ai_bs)
    ai_model = AIModel(D, 16, 2).to(device)
    opt_ai = optim.Adam(ai_model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    for ep in range(15):
        ai_model.train()
        for batch in ai_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = ai_model(batch["x"])
            loss = ce(out, batch["y"])
            opt_ai.zero_grad()
            loss.backward()
            opt_ai.step()
    # teacher probs
    ai_model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(np.vstack([X_train, X_val, X_test])).float().to(device)
        logits_all = ai_model(X_all)
        probs_all = torch.softmax(logits_all, dim=1).cpu().numpy()
    p_tr = probs_all[: len(X_train)]
    p_va = probs_all[len(X_train) : len(X_train) + len(X_val)]
    p_te = probs_all[-len(X_test) :]
    f_tr, f_va, f_te = p_tr.argmax(1), p_va.argmax(1), p_te.argmax(1)

    # bias-awareness signal
    def bias_signal(probs):
        ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        return 1 - ent / np.log(probs.shape[1])

    b_tr, b_va, b_te = bias_signal(p_tr), bias_signal(p_va), bias_signal(p_te)

    # feature matrices
    X_base = (X_train, X_val, X_test)
    X_cont = (
        np.hstack([X_train, p_tr]),
        np.hstack([X_val, p_va]),
        np.hstack([X_test, p_te]),
    )
    X_bias = (
        np.hstack([X_train, b_tr[:, None]]),
        np.hstack([X_val, b_va[:, None]]),
        np.hstack([X_test, b_te[:, None]]),
    )
    X_dual = (
        np.hstack([X_train, p_tr, b_tr[:, None]]),
        np.hstack([X_val, p_va, b_va[:, None]]),
        np.hstack([X_test, p_te, b_te[:, None]]),
    )

    for usr_bs in usr_batch_sizes:
        key = f"ai_bs_{ai_bs}_user_bs_{usr_bs}"
        scenarios = {
            "CE_hard_labels": {
                "feat": X_base,
                "hard": (f_tr, f_va, f_te),
                "use_soft": False,
                "input_dim": D,
            },
            "soft_label_distillation": {
                "feat": X_cont,
                "soft": (p_tr, p_va, p_te),
                "hard": (f_tr, f_va, f_te),
                "use_soft": True,
                "input_dim": D + 2,
            },
            "bias_awareness": {
                "feat": X_bias,
                "hard": (f_tr, f_va, f_te),
                "use_soft": False,
                "input_dim": D + 1,
            },
            "dual_channel": {
                "feat": X_dual,
                "soft": (p_tr, p_va, p_te),
                "hard": (f_tr, f_va, f_te),
                "use_soft": True,
                "input_dim": D + 3,
            },
        }

        for name, sc in scenarios.items():
            # dataset setup
            if sc["use_soft"]:
                tr_ds = SoftLabelDS(sc["feat"][0], sc["soft"][0], sc["hard"][0])
                va_ds = SoftLabelDS(sc["feat"][1], sc["soft"][1], sc["hard"][1])
                te_ds = SoftLabelDS(sc["feat"][2], sc["soft"][2], sc["hard"][2])
            else:
                tr_ds = UserDS(sc["feat"][0], sc["hard"][0])
                va_ds = UserDS(sc["feat"][1], sc["hard"][1])
                te_ds = UserDS(sc["feat"][2], sc["hard"][2])

            tr_ld = DataLoader(tr_ds, batch_size=usr_bs, shuffle=True)
            va_ld = DataLoader(va_ds, batch_size=usr_bs, shuffle=False)

            # model setup
            um = UserModel(sc["input_dim"], 8, 2).to(device)
            opt_u = optim.Adam(um.parameters(), lr=1e-2)
            if sc["use_soft"]:
                loss_fn = nn.KLDivLoss(reduction="batchmean")
            else:
                loss_fn = nn.CrossEntropyLoss()

            # tracking
            tr_acc, va_acc = [], []
            tr_ls, va_ls = [], []
            align_rates, align_hist = [], []

            # train epochs
            for epoch in range(1, 21):
                um.train()
                t_loss = corr = n = 0
                for batch in tr_ld:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = um(batch["feat"])
                    if sc["use_soft"]:
                        logp = F.log_softmax(out, dim=1)
                        loss = loss_fn(logp, batch["soft"])
                        true = batch["hard"]
                    else:
                        loss = loss_fn(out, batch["label"])
                        true = batch["label"]
                    opt_u.zero_grad()
                    loss.backward()
                    opt_u.step()
                    t_loss += loss.item() * batch["feat"].size(0)
                    preds = out.argmax(1)
                    corr += (preds == true).sum().item()
                    n += batch["feat"].size(0)
                tr_ls.append(t_loss / n)
                tr_acc.append(corr / n)

                um.eval()
                with torch.no_grad():
                    x_va = torch.from_numpy(sc["feat"][1]).float().to(device)
                    out_va = um(x_va)
                    if sc["use_soft"]:
                        logp_va = F.log_softmax(out_va, dim=1)
                        v_loss = loss_fn(
                            logp_va, torch.from_numpy(sc["soft"][1]).float().to(device)
                        ).item()
                        true_va = torch.from_numpy(sc["hard"][1]).long().to(device)
                    else:
                        v_loss = loss_fn(
                            out_va, torch.from_numpy(sc["hard"][1]).long().to(device)
                        ).item()
                        true_va = torch.from_numpy(sc["hard"][1]).long().to(device)
                    preds_va = out_va.argmax(1)
                    va_ls.append(v_loss)
                    va_acc.append((preds_va == true_va).float().mean().item())

                    # alignment
                    user_p = F.softmax(out_va, dim=1).cpu().numpy()
                    model_p = p_va
                    jsd = js_div(user_p, model_p) / np.log(2)
                    mean_align = (1 - jsd).mean()
                    align_hist.append(mean_align)
                    # safe slope computation
                    if len(align_hist) > 1:
                        try:
                            rate = np.polyfit(
                                np.arange(len(align_hist)), align_hist, 1
                            )[0]
                        except np.linalg.LinAlgError:
                            rate = 0.0
                    else:
                        rate = 0.0
                    align_rates.append(rate)

                print(f"{name} {key} Epoch {epoch}: validation_loss = {v_loss:.4f}")

            # test
            um.eval()
            x_te = torch.from_numpy(sc["feat"][2]).float().to(device)
            with torch.no_grad():
                preds_te = um(x_te).argmax(1).cpu().numpy()
            experiment_data[name][key] = {
                "metrics": {
                    "train": np.array(tr_acc),
                    "val": np.array(va_acc),
                    "alignment_rate": np.array(align_rates),
                },
                "losses": {"train": np.array(tr_ls), "val": np.array(va_ls)},
                "predictions": preds_te,
                "ground_truth": np.array(sc["hard"][2]),
            }

# save all results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
