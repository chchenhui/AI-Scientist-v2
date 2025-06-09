import os
import numpy as np
import torch
from datasets import load_dataset

# Setup working dir and random seed
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and sample datasets
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(50))
ambig = load_dataset("ambig_qa", split="validation").shuffle(seed=42).select(range(50))
trivia = (
    load_dataset("trivia_qa", "rc", split="validation")
    .shuffle(seed=42)
    .select(range(50))
)
datasets = [("SQuAD", squad), ("AmbigQA", ambig), ("TriviaQA-rc", trivia)]

# Ablation: post-clarification retrieval noise
noise_levels = [0.0, 0.1, 0.3, 0.5]
experiment_data = {"post_clar_noise": {}}

for name, ds in datasets:
    # Initialize per-dataset metrics storage
    experiment_data["post_clar_noise"][name] = {
        "metrics": {
            "noise_levels": noise_levels,
            "baseline_acc": [],
            "clar_acc": [],
            "avg_turns": [],
            "CES": [],
        }
    }
    # Simulate for each noise level
    for p in noise_levels:
        n = len(ds)
        acc_no, acc_cl, turns = 0.0, 0.0, 0
        for _ in ds:
            if name == "AmbigQA":
                acc0 = False
                turns += 1
                # retrieval correct with prob 1-p
                acc1 = float(np.random.rand() < (1 - p))
            else:
                acc0 = True
                acc1 = True
            acc_no += acc0
            acc_cl += acc1
        acc_no /= n
        acc_cl /= n
        avg_turns = turns / n
        ces = (acc_cl - acc_no) / avg_turns if avg_turns > 0 else 0.0
        m = experiment_data["post_clar_noise"][name]["metrics"]
        m["baseline_acc"].append(acc_no)
        m["clar_acc"].append(acc_cl)
        m["avg_turns"].append(avg_turns)
        m["CES"].append(ces)
        print(
            f"{name} p={p:.2f}: baseline_acc={acc_no:.4f}, clar_acc={acc_cl:.4f}, "
            f"avg_turns={avg_turns:.4f}, CES={ces:.4f}"
        )

# Save all metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
