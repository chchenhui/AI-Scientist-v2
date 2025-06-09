import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset, DataLoader

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 32
num_epochs = 3
max_len = 50
vocab_size = 1000


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_labels):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.fc = nn.Linear(emb_dim, n_labels)

    def forward(self, x):
        e = self.emb(x)  # [B, L, E]
        e = e.mean(dim=1)  # [B, E]
        return self.fc(e)  # [B, n_labels]


class TextDataset(Dataset):
    def __init__(self, texts, labels, token2id):
        self.data = []
        self.oov_flags = []
        for txt, lab in zip(texts, labels):
            toks = txt.split()
            ids = [token2id.get(t, 0) for t in toks]
            self.oov_flags.append(any(i == 0 for i in ids))
            ids = ids[:max_len] + [0] * (max_len - len(ids[:max_len]))
            self.data.append((torch.tensor(ids, dtype=torch.long), int(lab)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


experiment_data = {}

for ds_name in ["imdb", "ag_news", "yelp_polarity"]:
    raw = load_dataset(ds_name)
    # Shuffle before selecting to get balanced labels
    train_split = raw["train"].shuffle(seed=42).select(range(500))
    val_split = raw["test"].shuffle(seed=42).select(range(100))

    train_texts = [ex["text"] for ex in train_split]
    train_labels = [ex["label"] for ex in train_split]
    val_texts = [ex["text"] for ex in val_split]
    val_labels = [ex["label"] for ex in val_split]

    # Build vocabulary
    cnt = Counter()
    for t in train_texts:
        cnt.update(t.split())
    common = [w for w, _ in cnt.most_common(vocab_size)]
    token2id = {w: i + 1 for i, w in enumerate(common)}

    # Datasets & loaders
    train_ds = TextDataset(train_texts, train_labels, token2id)
    val_ds = TextDataset(val_texts, val_labels, token2id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Get the correct number of classes from metadata
    label_feat = raw["train"].features["label"]
    if isinstance(label_feat, ClassLabel):
        n_labels = label_feat.num_classes
    else:
        n_labels = len(set(train_labels + val_labels))

    # Model, optimizer, criterion
    model = TextClassifier(vocab_size, 64, n_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Metrics storage
    train_losses, val_losses, val_accs, cer_rates = [], [], [], []
    all_preds, all_gts = [], []

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_losses.append(total_loss / len(train_ds))

        # Validation
        model.eval()
        total_vl, correct = 0, 0
        preds, gts = [], []
        flags = val_ds.oov_flags
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_vl += loss.item() * x.size(0)
                p = logits.argmax(dim=1)
                correct += (p == y).sum().item()
                preds.extend(p.cpu().tolist())
                gts.extend(y.cpu().tolist())
        val_loss = total_vl / len(val_ds)
        val_acc = correct / len(val_ds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Constraint Effectiveness Rate
        constraint_count = sum(flags)
        effective = sum(
            1 for i, (f, pr, gt) in enumerate(zip(flags, preds, gts)) if f and pr == gt
        )
        cer = effective / constraint_count if constraint_count > 0 else 0.0
        cer_rates.append(cer)

        # Store epoch predictions/ground truth
        all_preds.append(preds)
        all_gts.append(gts)

        print(
            f"{ds_name} Epoch {epoch}: validation_loss = {val_loss:.4f}, constraint_effectiveness_rate = {cer:.4f}"
        )

    experiment_data[ds_name] = {
        "losses": {"train": train_losses, "val": val_losses},
        "metrics": {"val_acc": val_accs, "constraint_effectiveness_rate": cer_rates},
        "predictions": all_preds,
        "ground_truth": all_gts,
    }

# Save all data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Saved experiment_data.npy")
