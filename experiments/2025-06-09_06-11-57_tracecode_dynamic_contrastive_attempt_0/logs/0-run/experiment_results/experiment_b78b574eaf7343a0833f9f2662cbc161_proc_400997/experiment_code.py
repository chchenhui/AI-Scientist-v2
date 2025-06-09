import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# synthetic codes & traces
codes = []
for c in range(1, 11):
    codes.append(f"def f(x): return x+{c}")
    codes.append(f"def f(x): return {c}+x")
input_set = np.random.randint(-10, 10, 100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in input_set))
trace_to_indices = {}
for idx, trace in enumerate(traces):
    trace_to_indices.setdefault(trace, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for i in idxs:
        index_to_gid[i] = gid

# char‐level encoding
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi[c] for c in s] + [0] * (max_len - len(s)) for s in codes]
)


# dataset & splits
class CodeDataset(Dataset):
    def __init__(self, encoded, group_to_indices, index_to_gid):
        self.encoded, self.group_to_indices, self.index_to_gid = (
            encoded,
            group_to_indices,
            index_to_gid,
        )

    def __len__(self):
        return len(self.index_to_gid)

    def __getitem__(self, idx):
        a = self.encoded[idx]
        gid = self.index_to_gid[idx]
        p = idx
        while p == idx:
            p = random.choice(self.group_to_indices[gid])
        neg_gid = random.choice([g for g in self.group_to_indices if g != gid])
        n = random.choice(self.group_to_indices[neg_gid])
        return a, self.encoded[p], self.encoded[n]


dataset = CodeDataset(encoded, group_to_indices, index_to_gid)
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)


# encoder with optional bidirectionality
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64, bidirectional=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bidirectional = bidirectional
        nh = hidden // (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            embed_dim, nh, batch_first=True, bidirectional=bidirectional
        )

    def forward(self, x):
        x = self.embed(x)
        h, _ = self.lstm(x)  # h: (B, T, D*num_dirs)
        # get final hidden states
        _, (hn, _) = self.lstm(x)
        if self.bidirectional:
            h1, h2 = hn[0], hn[1]
            return torch.cat([h1, h2], dim=1)
        else:
            return hn.squeeze(0)


# ablation study
variants = ["unidirectional", "bidirectional"]
EPOCH_LIST = [10, 30, 50]
experiment_data = {
    "bidirectional_lstm_ablation": {"synthetic": {v: {} for v in variants}}
}

for variant in variants:
    is_bi = variant == "bidirectional"
    for E in EPOCH_LIST:
        model = CodeEncoder(len(stoi), embed_dim=64, hidden=64, bidirectional=is_bi).to(
            device
        )
        optimizer = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0)

        data = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        for epoch in range(E):
            model.train()
            tr_loss = 0.0
            for a, p, n in train_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                ea, ep, en = model(a), model(p), model(n)
                l = loss_fn(ea, ep, en)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                tr_loss += l.item()
            data["losses"]["train"].append(tr_loss / len(train_loader))

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for a, p, n in val_loader:
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    vl += loss_fn(model(a), model(p), model(n)).item()
            data["losses"]["val"].append(vl / len(val_loader))

            # retrieval acc
            with torch.no_grad():
                emb_all = model(encoded.to(device))
                emb_n = F.normalize(emb_all, dim=1)
                sims = emb_n @ emb_n.T

                def comp_acc(indices):
                    corr = 0
                    for i in indices:
                        s = sims[i].clone()
                        s[i] = -1e9
                        top1 = torch.argmax(s).item()
                        if index_to_gid[top1] == index_to_gid[i]:
                            corr += 1
                    return corr / len(indices)

                data["metrics"]["train"].append(comp_acc(train_indices))
                data["metrics"]["val"].append(comp_acc(val_indices))

        # final predictions
        model.eval()
        with torch.no_grad():
            emb_all = model(encoded.to(device))
            emb_n = F.normalize(emb_all, dim=1)
            sims = emb_n @ emb_n.T
            for i in val_indices:
                s = sims[i].clone()
                s[i] = -1e9
                top1 = torch.argmax(s).item()
                data["predictions"].append(index_to_gid[top1])
                data["ground_truth"].append(index_to_gid[i])

        # convert to numpy arrays
        data_np = {
            "metrics": {
                "train": np.array(data["metrics"]["train"]),
                "val": np.array(data["metrics"]["val"]),
            },
            "losses": {
                "train": np.array(data["losses"]["train"]),
                "val": np.array(data["losses"]["val"]),
            },
            "predictions": np.array(data["predictions"]),
            "ground_truth": np.array(data["ground_truth"]),
        }
        experiment_data["bidirectional_lstm_ablation"]["synthetic"][variant][
            E
        ] = data_np
        print(f"{variant} E={E} final val acc={data_np['metrics']['val'][-1]:.4f}")

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
