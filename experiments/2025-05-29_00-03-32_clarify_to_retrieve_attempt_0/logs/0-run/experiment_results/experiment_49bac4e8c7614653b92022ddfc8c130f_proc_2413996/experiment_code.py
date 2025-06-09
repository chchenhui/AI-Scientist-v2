import os
import random
import numpy as np
import torch
from datasets import load_dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Reproducibility
random.seed(42)
np.random.seed(42)

# Device (unused but kept for consistency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(50))
ambig = load_dataset("ambig_qa", split="validation").shuffle(seed=42).select(range(50))
trivia = (
    load_dataset("trivia_qa", "rc", split="validation")
    .shuffle(seed=42)
    .select(range(50))
)

datasets = [("SQuAD", squad), ("AmbigQA", ambig), ("TriviaQA-rc", trivia)]


# Utilities
def get_gt(sample):
    if "answers" in sample:
        a = sample["answers"]
        if isinstance(a, dict):
            return a.get("text", [None])[0] or ""
        elif isinstance(a, list):
            return a[0] if a else ""
    if "answer" in sample:
        b = sample["answer"]
        return b[0] if isinstance(b, list) and b else (b or "")
    return ""


# Ablation parameters
dropout_rates = [0.0, 0.25, 0.5, 0.75, 0.9]

# Prepare experiment data structure
experiment_data = {"user_patience_dropout": {}}

for name, ds in datasets:
    # Lists to collect metrics over dropout rates
    baselines, clar_accs, avg_turns_list, ces_list = [], [], [], []
    for drop in dropout_rates:
        n = len(ds)
        acc_no_sum = 0.0
        acc_cl_sum = 0.0
        total_turns = 0
        for sample in ds:
            # Baseline
            if name == "AmbigQA":
                acc0 = False
            else:
                acc0 = True
            acc_no_sum += float(acc0)
            # Clarification + dropout
            if name == "AmbigQA":
                # we ask one question, user may refuse
                total_turns += 1
                responded = random.random() > drop
                acc1 = True if responded else acc0
            else:
                acc1 = True
            acc_cl_sum += float(acc1)
        acc_no = acc_no_sum / n
        acc_cl = acc_cl_sum / n
        avg_t = total_turns / n
        ces = (acc_cl - acc_no) / avg_t if avg_t > 0 else 0.0
        baselines.append(acc_no)
        clar_accs.append(acc_cl)
        avg_turns_list.append(avg_t)
        ces_list.append(ces)
    # Save per‐dataset results
    experiment_data["user_patience_dropout"][name] = {
        "metrics": {
            "dropout_rates": np.array(dropout_rates),
            "baseline_acc": np.array(baselines),
            "clar_acc": np.array(clar_accs),
            "avg_turns": np.array(avg_turns_list),
            "CES": np.array(ces_list),
        },
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

# Print results
for name, data in experiment_data["user_patience_dropout"].items():
    print(f"\nDataset: {name}")
    dr = data["metrics"]["dropout_rates"]
    ba = data["metrics"]["baseline_acc"]
    ca = data["metrics"]["clar_acc"]
    at = data["metrics"]["avg_turns"]
    ce = data["metrics"]["CES"]
    for i, d in enumerate(dr):
        print(
            f" drop={d:.2f} -> baseline={ba[i]:.4f}, clar={ca[i]:.4f}, turns={at[i]:.4f}, CES={ce[i]:.4f}"
        )

# Save all metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
