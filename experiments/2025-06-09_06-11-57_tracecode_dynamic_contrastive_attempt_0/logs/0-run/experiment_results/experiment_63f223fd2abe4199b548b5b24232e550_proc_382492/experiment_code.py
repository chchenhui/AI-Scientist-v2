import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1) Build synthetic code corpus with simple variants
base_codes = [
    "def f(x): return x+1",
    "def g(x): return x*2",
    "def h(x): return x if x>0 else -x",
    "def p(x): return x*x",
]
code_list = []
for code in base_codes:
    name = code.split()[1].split("(")[0]
    expr = code.split("return ")[1]
    variant = f"def {name}(x):\n    tmp = {expr}\n    return tmp"
    code_list += [code, variant]

# 2) Execute each snippet on random inputs to get dynamic traces
inputs = [random.randint(-10, 10) for _ in range(20)]
func_objs = []
for code in code_list:
    env = {}
    exec(code, {}, env)
    for v in env.values():
        if callable(v):
            func_objs.append(v)
            break
traces = [[f(x) for x in inputs] for f in func_objs]

# 3) Char-level tokenization
chars = sorted({c for s in code_list for c in s})
char2idx = {c: i + 1 for i, c in enumerate(chars)}
max_len = max(len(s) for s in code_list)
tokens_list = []
for s in code_list:
    seq = [char2idx[c] for c in s] + [0] * (max_len - len(s))
    tokens_list.append(seq)

# 4) Dataset of triplets (anchor, positive, negative)
pairs = [(i, i + 1) for i in range(0, len(code_list), 2)]


class TraceDataset(Dataset):
    def __init__(self, tokens, pairs):
        self.tokens = tokens
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        # random negative index
        neg = random.choice([k for k in range(len(self.tokens)) if k not in (i, j)])
        return (
            torch.LongTensor(self.tokens[i]),
            torch.LongTensor(self.tokens[j]),
            torch.LongTensor(self.tokens[neg]),
        )


def collate(batch):
    A = torch.stack([a for a, _, _ in batch])
    P = torch.stack([p for _, p, _ in batch])
    N = torch.stack([n for _, _, n in batch])
    return A, P, N


dataset = TraceDataset(tokens_list, pairs)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)


# 5) Model: Transformer encoder + pooling
class CodeEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, L)
        emb = self.embedding(x).transpose(0, 1)  # L, B, D
        out = self.transformer(emb).transpose(0, 1)  # B, L, D
        return self.pool(out.transpose(1, 2)).squeeze(-1)  # B, D


model = CodeEncoder(len(char2idx) + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 6) Experiment logging structure
experiment_data = {
    "trace_dataset": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# 7) Training loop with triplet loss and retrieval eval
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for A, P, N in loader:
        A, P, N = A.to(device), P.to(device), N.to(device)
        eA, eP, eN = model(A), model(P), model(N)
        loss = F.triplet_margin_loss(eA, eP, eN, margin=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * A.size(0)
    train_loss = total_loss / len(dataset)
    # validation pass (same data here for simplicity)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for A, P, N in loader:
            A, P, N = A.to(device), P.to(device), N.to(device)
            eA, eP, eN = model(A), model(P), model(N)
            val_loss += F.triplet_margin_loss(eA, eP, eN, margin=1.0).item() * A.size(0)
    val_loss /= len(dataset)
    # retrieval accuracy
    with torch.no_grad():
        all_toks = torch.LongTensor(tokens_list).to(device)
        emb = F.normalize(model(all_toks), dim=1)
        correct = 0
        for i, j in pairs:
            sims = (emb[i : i + 1] @ emb.t()).squeeze(0)
            sims[i] = -1e9
            pred = sims.argmax().item()
            if pred == j:
                correct += 1
    acc = correct / len(pairs)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}, retrieval_acc = {acc:.4f}")
    experiment_data["trace_dataset"]["losses"]["train"].append(train_loss)
    experiment_data["trace_dataset"]["losses"]["val"].append(val_loss)
    experiment_data["trace_dataset"]["metrics"]["train"].append(acc)
    experiment_data["trace_dataset"]["metrics"]["val"].append(acc)

# 8) Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
