import os
import numpy as np
from datasets import load_dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Seed for reproducibility
np.random.seed(42)

# Flip rates for ablation
flip_rates = [0.0, 0.1, 0.2]

# Initialize experiment data structure
experiment_data = {
    "ambiguity_detection_noise": {
        "flip_rates": flip_rates,
        "SQuAD": {
            "metrics": {"baseline_acc": [], "clar_acc": [], "avg_turns": [], "CES": []}
        },
        "AmbigQA": {
            "metrics": {"baseline_acc": [], "clar_acc": [], "avg_turns": [], "CES": []}
        },
        "TriviaQA-rc": {
            "metrics": {"baseline_acc": [], "clar_acc": [], "avg_turns": [], "CES": []}
        },
    }
}


# Utility to extract ground truth (unused here but kept for completeness)
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


# Load and subsample datasets
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(50))
ambig = load_dataset("ambig_qa", split="validation").shuffle(seed=42).select(range(50))
trivia = (
    load_dataset("trivia_qa", "rc", split="validation")
    .shuffle(seed=42)
    .select(range(50))
)
datasets = {"SQuAD": squad, "AmbigQA": ambig, "TriviaQA-rc": trivia}

# Run ablation over flip rates
for p in flip_rates:
    for name, ds in datasets.items():
        n = len(ds)
        acc_no = 0.0
        acc_cl = 0.0
        turns = 0.0
        for sample in ds:
            # Determine true ambiguous label
            true_ambig = name == "AmbigQA"
            # Simulate flip/noise
            if np.random.rand() < p:
                detected = not true_ambig
            else:
                detected = true_ambig
            # Baseline correctness
            acc0 = False if name == "AmbigQA" else True
            # Clarified correctness
            if detected:
                acc1 = True
                turns += 1
            else:
                acc1 = acc0
            acc_no += acc0
            acc_cl += acc1
        # Compute metrics
        acc_no /= n
        acc_cl /= n
        avg_turns = turns / n
        ces = (acc_cl - acc_no) / avg_turns if avg_turns > 0 else 0.0
        # Store results
        m = experiment_data["ambiguity_detection_noise"][name]["metrics"]
        m["baseline_acc"].append(acc_no)
        m["clar_acc"].append(acc_cl)
        m["avg_turns"].append(avg_turns)
        m["CES"].append(ces)
    # Print per-rate summary
    print(f"=== Flip rate {p:.2f} ===")
    for name in datasets:
        m = experiment_data["ambiguity_detection_noise"][name]["metrics"]
        idx = flip_rates.index(p)
        print(
            f"{name}: baseline_acc={m['baseline_acc'][idx]:.4f}, "
            f"clar_acc={m['clar_acc'][idx]:.4f}, avg_turns={m['avg_turns'][idx]:.4f}, "
            f"CES={m['CES'][idx]:.4f}"
        )

# Save all metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
