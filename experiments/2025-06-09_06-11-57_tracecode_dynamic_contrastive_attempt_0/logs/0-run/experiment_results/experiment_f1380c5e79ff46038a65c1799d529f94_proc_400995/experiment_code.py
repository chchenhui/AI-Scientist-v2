import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# synthetic data
codes = []
for c in range(1, 11):
    codes.append(f"def f(x): return x+{c}")
    codes.append(f"def f(x): return {c}+x")
inp = np.random.randint(-10, 10, size=100)
traces = []
for code in codes:
    env = {}
    exec(code, env)
    f = env["f"]
    traces.append(tuple(f(int(x)) for x in inp))

trace_to_idxs = {}
for i, t in enumerate(traces):
    trace_to_idxs.setdefault(t, []).append(i)
group_to_idxs = {g: idxs for g, idxs in enumerate(trace_to_idxs.values())}
index_to_gid = [None] * len(codes)
for g, idxs in group_to_idxs.items():
    for i in idxs:
        index_to_gid[i] = g

# encode chars
vocab = sorted(set("".join(codes)))
stoi = {c: i + 1 for i, c in enumerate(vocab)}
stoi["PAD"] = 0
max_len = max(len(s) for s in codes)
encoded = torch.LongTensor(
    [[stoi[c] for c in s] + [0] * (max_len - len(s)) for s in codes]
)


# dataset
class CodeDataset(Dataset):
    def __init__(self, enc, group_to_idxs, index_to_gid):
        self.enc = enc
        self.g2i = group_to_idxs
        self.i2g = index_to_gid

    def __len__(self):
        return len(self.i2g)

    def __getitem__(self, idx):
        a = self.enc[idx]
        g = self.i2g[idx]
        pos = idx
        while pos == idx:
            pos = random.choice(self.g2i[g])
        neg_g = random.choice([x for x in self.g2i if x != g])
        neg = random.choice(self.g2i[neg_g])
        return a, self.enc[pos], self.enc[neg]


dataset = CodeDataset(encoded, group_to_idxs, index_to_gid)
all_gs = list(group_to_idxs.keys())
random.shuffle(all_gs)
split = int(0.8 * len(all_gs))
train_gs, val_gs = all_gs[:split], all_gs[split:]
train_idxs = [i for g in train_gs for i in group_to_idxs[g]]
val_idxs = [i for g in val_gs for i in group_to_idxs[g]]
train_loader = DataLoader(Subset(dataset, train_idxs), batch_size=8, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idxs), batch_size=8, shuffle=False)


# CNN encoder
class CodeEncoderCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        hidden=64,
        kernel_sizes=[3, 4, 5],
        num_filters=64,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=k
                )
                for k in kernel_sizes
            ]
        )
        self.fc = nn.Linear(num_filters * len(kernel_sizes), hidden)

    def forward(self, x):
        x = self.embed(x)  # (B, L, E)
        x = x.transpose(1, 2)  # (B, E, L)
        outs = [F.relu(conv(x)) for conv in self.convs]
        pools = [F.max_pool1d(o, o.size(2)).squeeze(2) for o in outs]
        cat = torch.cat(pools, dim=1)
        return self.fc(cat)


# experiment scaffolding
EPOCH_LIST = [10, 30, 50]
experiment_data = {"CNN_ENCODER_ABLATION": {"synthetic": {}}}

for E in EPOCH_LIST:
    model = CodeEncoderCNN(len(stoi), 64, 64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.TripletMarginLoss(margin=1.0)
    data = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
        "param_count": sum(p.numel() for p in model.parameters()),
    }

    for epoch in range(E):
        model.train()
        tr_loss = 0.0
        for a, p, n in train_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ea, ep, en = model(a), model(p), model(n)
            loss = loss_fn(ea, ep, en)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)
        data["losses"]["train"].append(tr_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for a, p, n in val_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                val_loss += loss_fn(model(a), model(p), model(n)).item()
        val_loss /= len(val_loader)
        data["losses"]["val"].append(val_loss)

        with torch.no_grad():
            emb = model(encoded.to(device))
            emb_n = F.normalize(emb, dim=1)
            sims = emb_n @ emb_n.T

            def acc(idx_list):
                c = 0
                for i in idx_list:
                    s = sims[i].clone()
                    s[i] = -1e9
                    top1 = torch.argmax(s).item()
                    if index_to_gid[top1] == index_to_gid[i]:
                        c += 1
                return c / len(idx_list)

            data["metrics"]["train"].append(acc(train_idxs))
            data["metrics"]["val"].append(acc(val_idxs))

    # final predictions
    model.eval()
    with torch.no_grad():
        emb = model(encoded.to(device))
        emb_n = F.normalize(emb, dim=1)
        sims = emb_n @ emb_n.T
        for i in val_idxs:
            s = sims[i].clone()
            s[i] = -1e9
            top1 = torch.argmax(s).item()
            data["predictions"].append(index_to_gid[top1])
            data["ground_truth"].append(index_to_gid[i])

    experiment_data["CNN_ENCODER_ABLATION"]["synthetic"][E] = data
    print(
        f"Done CNN ablation EPOCHS={E}, final val acc={data['metrics']['val'][-1]:.4f}"
    )

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
