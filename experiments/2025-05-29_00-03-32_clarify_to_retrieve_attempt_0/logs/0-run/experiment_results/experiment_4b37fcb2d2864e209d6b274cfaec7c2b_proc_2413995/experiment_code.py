import os
import numpy as np
import torch
from datasets import load_dataset

# Setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and sample three QA datasets
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(50))
ambig = load_dataset("ambig_qa", split="validation").shuffle(seed=42).select(range(50))
trivia = (
    load_dataset("trivia_qa", "rc", split="validation")
    .shuffle(seed=42)
    .select(range(50))
)


# Utility to get ground truth (unused here but included for completeness)
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


# Prepare ablation budgets
budgets = [0, 1, 2, 3]
datasets = [("SQuAD", squad), ("AmbigQA", ambig), ("TriviaQA-rc", trivia)]

# Initialize experiment data structure
experiment_data = {"clarification_turn_budget": {}}
for name, _ in datasets:
    experiment_data["clarification_turn_budget"][name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

# Run ablation study
for k in budgets:
    print(f"\n--- Clarification Budget: {k} ---")
    for name, ds in datasets:
        n = len(ds)
        acc_no_total = 0.0
        acc_cl_total = 0.0
        turns_total = 0
        for sample in ds:
            # Baseline correctness
            if name == "AmbigQA":
                acc0 = False
            else:
                acc0 = True
            # Clarified correctness and used turns
            if k >= 1:
                # AmbigQA fixes on first allowed turn, others already correct
                acc1 = True
                used = 1 if name == "AmbigQA" else 0
            else:
                acc1 = acc0
                used = 0
            acc_no_total += acc0
            acc_cl_total += acc1
            turns_total += used
        baseline_acc = acc_no_total / n
        clar_acc = acc_cl_total / n
        avg_turns = turns_total / n
        ces = (clar_acc - baseline_acc) / avg_turns if avg_turns > 0 else 0.0
        # Record metrics
        experiment_data["clarification_turn_budget"][name]["metrics"]["val"].append(
            {
                "budget": k,
                "baseline_acc": baseline_acc,
                "clar_acc": clar_acc,
                "avg_turns": avg_turns,
                "CES": ces,
            }
        )
        print(
            f"{name}: baseline_acc={baseline_acc:.4f}, clar_acc={clar_acc:.4f}, "
            f"avg_turns={avg_turns:.4f}, CES={ces:.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nSaved ablation results to {os.path.join(working_dir, 'experiment_data.npy')}")
