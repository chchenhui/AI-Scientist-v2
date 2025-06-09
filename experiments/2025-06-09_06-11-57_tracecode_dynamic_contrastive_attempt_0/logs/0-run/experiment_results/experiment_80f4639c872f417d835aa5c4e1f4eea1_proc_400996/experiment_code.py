import os
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# reproducibility & setup
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# generate synthetic snippets and semantic traces
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

# group by semantic equivalence
trace_to_indices = {}
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for idx in idxs:
        index_to_gid[idx] = gid

# generate one unseen random renaming per snippet
renamed_codes = []
for s in codes:
    func = f"g{random.randint(100,999)}"
    var = f"y{random.randint(100,999)}"
    s2 = re.sub(r"\bdef\s+f\(", f"def {func}(", s)
    s3 = re.sub(r"\bx\b", var, s2)
    renamed_codes.append(s3)

# build joint vocab over original + renamed
all_text = "".join(codes + renamed_codes)
vocab = sorted(set(all_text))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in (codes + renamed_codes))


# helper to encode and pad
def encode_list(strs):
    arr = []
    for s in strs:
        seq = [stoi.get(c, 0) for c in s] + [0] * (max_len - len(s))
        arr.append(seq)
    return torch.LongTensor(arr)


encoded = encode_list(codes)
encoded_renamed = encode_list(renamed_codes)

# train/val split on groups
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]


# dataset & loaders
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


dataset = CodeDataset(encoded, group_to_indices, index_to_gid)
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)


# model
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# ablation experiment
EPOCH_LIST = [10, 30, 50]
experiment_data = {"variable_renaming_invariance": {"synthetic": {}}}

for E in EPOCH_LIST:
    model = CodeEncoder(len(stoi), 64, 64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    data = {
        "metrics": {"train": [], "val": [], "rename": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "rename_predictions": [],
        "rename_ground_truth": [],
    }

    # train + epochwise eval
    for epoch in range(E):
        model.train()
        tot_tr = 0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ea, ep, en = model(a), model(p), model(n)
            loss = loss_fn(ea, ep, en)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_tr += loss.item()
        data["losses"]["train"].append(tot_tr / len(train_loader))

        model.eval()
        tot_val = 0
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                tot_val += loss_fn(model(a), model(p), model(n)).item()
        data["losses"]["val"].append(tot_val / len(val_loader))

        # retrieval acc on original
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            emb_norm = F.normalize(emb_all, dim=1)
            sims = emb_norm @ emb_norm.T

            def compute_acc(idxs):
                corr = 0
                for i in idxs:
                    s = sims[i].clone()
                    s[i] = -1e9
                    j = torch.argmax(s).item()
                    if index_to_gid[j] == index_to_gid[i]:
                        corr += 1
                return corr / len(idxs)

            tr_acc = compute_acc(train_indices)
            v_acc = compute_acc(val_indices)

        # retrieval acc on unseen renamings (val only)
        with torch.no_grad():
            emb_ren = model(encoded_renamed.to(device))
            ren_norm = F.normalize(emb_ren, dim=1)
            sims2 = ren_norm @ emb_norm.T
            corr2 = 0
            for i in val_indices:
                j = torch.argmax(sims2[i]).item()
                if index_to_gid[j] == index_to_gid[i]:
                    corr2 += 1
            ren_acc = corr2 / len(val_indices)

        data["metrics"]["train"].append(tr_acc)
        data["metrics"]["val"].append(v_acc)
        data["metrics"]["rename"].append(ren_acc)

    # final predictions
    model.eval()
    with torch.no_grad():
        emb_o = model(encoded.to(device))
        norm_o = F.normalize(emb_o, dim=1)
        sims_o = norm_o @ norm_o.T
        emb_r = model(encoded_renamed.to(device))
        norm_r = F.normalize(emb_r, dim=1)
        sims_r = norm_r @ norm_o.T

        for i in val_indices:
            # original
            s = sims_o[i].clone()
            s[i] = -1e9
            j = torch.argmax(s).item()
            data["predictions"].append(index_to_gid[j])
            data["ground_truth"].append(index_to_gid[i])
            # rename
            j2 = torch.argmax(sims_r[i]).item()
            data["rename_predictions"].append(index_to_gid[j2])
            data["rename_ground_truth"].append(index_to_gid[i])

    experiment_data["variable_renaming_invariance"]["synthetic"][E] = data
    print(
        f"Done E={E}: val_acc={data['metrics']['val'][-1]:.4f}, rename_acc={data['metrics']['rename'][-1]:.4f}"
    )

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
