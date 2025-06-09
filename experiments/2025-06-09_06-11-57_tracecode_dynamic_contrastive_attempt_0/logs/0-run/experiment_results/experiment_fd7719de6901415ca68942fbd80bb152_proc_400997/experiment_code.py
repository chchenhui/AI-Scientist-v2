import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

import random, re, ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# generate synthetic functions and traces
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

# group by trace
trace_to_indices = {}
for idx, tr in enumerate(traces):
    trace_to_indices.setdefault(tr, []).append(idx)
group_to_indices = {gid: idxs for gid, idxs in enumerate(trace_to_indices.values())}
index_to_gid = [None] * len(codes)
for gid, idxs in group_to_indices.items():
    for idx in idxs:
        index_to_gid[idx] = gid

# train/val split
all_gids = list(group_to_indices.keys())
random.shuffle(all_gids)
split = int(0.8 * len(all_gids))
train_gids, val_gids = all_gids[:split], all_gids[split:]
train_indices = [i for g in train_gids for i in group_to_indices[g]]
val_indices = [i for g in val_gids for i in group_to_indices[g]]


# Dataset
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


# Encoder
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)


# tokenizers
def char_tokenize(s):
    return list(s)


def subword_tokenize(s):
    return re.findall(r"\w+|[^\s\w]", s)


def ast_tokenize(s):
    tokens = []

    def visit(node):
        if isinstance(node, ast.Constant):
            tokens.append(f"Const_{node.value}")
        elif isinstance(node, ast.Name):
            tokens.append(f"Name_{node.id}")
        else:
            tokens.append(type(node).__name__)
            for _, v in ast.iter_fields(node):
                if isinstance(v, list):
                    for elt in v:
                        if isinstance(elt, ast.AST):
                            visit(elt)
                elif isinstance(v, ast.AST):
                    visit(v)

    visit(ast.parse(s))
    return tokens


schemes = {"char": char_tokenize, "subword": subword_tokenize, "ast": ast_tokenize}
EPOCH_LIST = [10, 30, 50]
experiment_data = {"tokenization_granularity": {"synthetic": {}}}

for scheme, tokfn in schemes.items():
    toks_list = [tokfn(s) for s in codes]
    vocab = sorted({t for toks in toks_list for t in toks})
    stoi = {t: i + 1 for i, t in enumerate(vocab)}
    stoi["PAD"] = 0
    max_len = max(len(toks) for toks in toks_list)
    encoded = torch.LongTensor(
        [[stoi[t] for t in toks] + [0] * (max_len - len(toks)) for toks in toks_list]
    )
    dataset = CodeDataset(encoded, group_to_indices, index_to_gid)
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=8, shuffle=True
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=8, shuffle=False)
    synth_data = {}
    for E in EPOCH_LIST:
        model = CodeEncoder(len(stoi), 64, 64).to(device)
        opt = Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        data = {
            "metrics": {
                "train_acc": [],
                "val_acc": [],
                "train_alignment_gap": [],
                "val_alignment_gap": [],
            },
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }
        for epoch in range(E):
            # training
            model.train()
            tot_loss = 0.0
            sum_pos, sum_neg, count = 0.0, 0.0, 0
            for a, p, n in train_loader:
                a, p, n = a.to(device), p.to(device), n.to(device)
                ea, ep, en = model(a), model(p), model(n)
                loss = loss_fn(ea, ep, en)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_loss += loss.item()
                pos = F.cosine_similarity(ea, ep, dim=1)
                neg = F.cosine_similarity(ea, en, dim=1)
                sum_pos += pos.sum().item()
                sum_neg += neg.sum().item()
                count += pos.size(0)
            avg_train_loss = tot_loss / len(train_loader)
            data["losses"]["train"].append(avg_train_loss)
            data["metrics"]["train_alignment_gap"].append(
                sum_pos / count - sum_neg / count
            )
            # train acc
            model.eval()
            with torch.no_grad():
                emb = model(encoded.to(device))
                nm = F.normalize(emb, dim=1)
                sims = nm @ nm.T

                def acc(idxs):
                    cnt = 0
                    for i in idxs:
                        s = sims[i].clone()
                        s[i] = -1e9
                        top = torch.argmax(s).item()
                        if index_to_gid[top] == index_to_gid[i]:
                            cnt += 1
                    return cnt / len(idxs)

                data["metrics"]["train_acc"].append(acc(train_indices))
            # validation
            tv_loss = 0.0
            sum_pos_v, sum_neg_v, count_v = 0.0, 0.0, 0
            with torch.no_grad():
                for a, p, n in val_loader:
                    a, p, n = a.to(device), p.to(device), n.to(device)
                    ea, ep, en = model(a), model(p), model(n)
                    tv_loss += loss_fn(ea, ep, en).item()
                    pos_v = F.cosine_similarity(ea, ep, dim=1)
                    neg_v = F.cosine_similarity(ea, en, dim=1)
                    sum_pos_v += pos_v.sum().item()
                    sum_neg_v += neg_v.sum().item()
                    count_v += pos_v.size(0)
            avg_val_loss = tv_loss / len(val_loader)
            data["losses"]["val"].append(avg_val_loss)
            data["metrics"]["val_alignment_gap"].append(
                sum_pos_v / count_v - sum_neg_v / count_v
            )
            print(f"Epoch {epoch+1}/{E}: validation_loss = {avg_val_loss:.4f}")
            with torch.no_grad():
                nm = F.normalize(model(encoded.to(device)), dim=1)
                sims = nm @ nm.T
                data["metrics"]["val_acc"].append(acc(val_indices))
        # final predictions
        model.eval()
        with torch.no_grad():
            emb = model(encoded.to(device))
            nm = F.normalize(emb, dim=1)
            sims = nm @ nm.T
            for i in val_indices:
                s = sims[i].clone()
                s[i] = -1e9
                top = torch.argmax(s).item()
                data["predictions"].append(index_to_gid[top])
                data["ground_truth"].append(index_to_gid[i])
        synth_data[E] = data
        print(
            f"Scheme={scheme}, E={E}, final val acc={data['metrics']['val_acc'][-1]:.4f}"
        )
    experiment_data["tokenization_granularity"]["synthetic"][scheme] = synth_data

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
