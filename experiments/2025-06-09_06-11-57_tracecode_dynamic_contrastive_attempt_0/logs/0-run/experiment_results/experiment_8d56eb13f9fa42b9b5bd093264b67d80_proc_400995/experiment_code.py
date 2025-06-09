import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# initialize experiment storage
experiment_data = {
    "dead_code_injection": {
        "synthetic_clean": {},
        "synthetic_injected": {},
    }
}

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# generate base synthetic code snippets and their traces
codes = []
for c in range(1, 11):
    codes.append(f"def f(x): return x+{c}")
    codes.append(f"def f(x): return {c}+x")
input_set = np.random.randint(-10, 10, size=100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in input_set))

# group snippets by identical trace (functional equivalence)
trace_to_indices = {}
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for idx in idxs:
        index_to_gid[idx] = gid

# train/validation split of groups
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]


# dead code injector utility
def inject_dead_code(code):
    # Turn "def f(x): return expr" into a multi-line with no-ops
    sig, expr = code.split(": return ")
    return (
        f"{sig}:\n"
        "    dummy = 0\n"
        "    for _ in range(1):\n"
        "        pass\n"
        f"    return {expr}"
    )


# dataset class
class CodeDataset(Dataset):
    def __init__(self, encoded, group_to_indices, index_to_gid):
        self.encoded = encoded
        self.group_to_indices = group_to_indices
        self.index_to_gid = index_to_gid

    def __len__(self):
        return len(self.index_to_gid)

    def __getitem__(self, idx):
        anchor = self.encoded[idx]
        gid = self.index_to_gid[idx]
        pos = idx
        while pos == idx:
            pos = random.choice(self.group_to_indices[gid])
        neg_gid = random.choice([g for g in self.group_to_indices if g != gid])
        neg = random.choice(self.group_to_indices[neg_gid])
        return anchor, self.encoded[pos], self.encoded[neg]


# model definition
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# hyperparameters
EPOCH_LIST = [10, 30, 50]

# run both clean and injected versions
for dataset_name in ["synthetic_clean", "synthetic_injected"]:
    # pick the code variants
    if dataset_name == "synthetic_clean":
        codes_use = codes
    else:
        codes_use = [inject_dead_code(c) for c in codes]

    # build vocabulary and encode
    vocab = sorted(set("".join(codes_use)))
    stoi = {c: i + 1 for i, c in enumerate(vocab)}
    stoi["PAD"] = 0
    max_len = max(len(s) for s in codes_use)
    encoded_list = []
    for s in codes_use:
        seq = [stoi[ch] for ch in s] + [0] * (max_len - len(s))
        encoded_list.append(seq)
    encoded_use = torch.LongTensor(encoded_list)

    # prepare data loaders
    dataset = CodeDataset(encoded_use, group_to_indices, index_to_gid)
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=8, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)

    # per-epoch-budget experiments
    for E in EPOCH_LIST:
        # init model & optimizer
        model = CodeEncoder(len(stoi), 64, 64).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        data = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        # train + validate
        for epoch in range(E):
            model.train()
            total_train_loss = 0.0
            for a, p, n in train_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                emb_a, emb_p, emb_n = model(a), model(p), model(n)
                loss = loss_fn(emb_a, emb_p, emb_n)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            data["losses"]["train"].append(total_train_loss / len(train_loader))

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for a, p, n in val_loader:
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    total_val_loss += loss_fn(model(a), model(p), model(n)).item()
            data["losses"]["val"].append(total_val_loss / len(val_loader))

            # retrieval accuracy
            with torch.no_grad():
                embs = model(encoded_use.to(device))
                embs = F.normalize(embs, dim=1)
                sims = embs @ embs.t()

                def compute_acc(idxs):
                    correct = 0
                    for i in idxs:
                        s = sims[i].clone()
                        s[i] = -float("inf")
                        top1 = torch.argmax(s).item()
                        if index_to_gid[top1] == index_to_gid[i]:
                            correct += 1
                    return correct / len(idxs)

                data["metrics"]["train"].append(compute_acc(train_indices))
                data["metrics"]["val"].append(compute_acc(val_indices))

        # final predictions on validation
        model.eval()
        with torch.no_grad():
            embs = model(encoded_use.to(device))
            embs = F.normalize(embs, dim=1)
            sims = embs @ embs.t()
            for i in val_indices:
                s = sims[i].clone()
                s[i] = -float("inf")
                top1 = torch.argmax(s).item()
                data["predictions"].append(index_to_gid[top1])
                data["ground_truth"].append(index_to_gid[i])

        experiment_data["dead_code_injection"][dataset_name][E] = data
        print(
            f"Dataset={dataset_name}, EPOCHS={E}, final val_acc={data['metrics']['val'][-1]:.4f}"
        )

# save everything
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
