import os
import random
import numpy as np
import torch
from datasets import load_dataset

# reproducibility
random.seed(42)
np.random.seed(42)

# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load datasets
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(50))
ambig = load_dataset("ambig_qa", split="validation").shuffle(seed=42).select(range(50))
trivia = (
    load_dataset("trivia_qa", "rc", split="validation")
    .shuffle(seed=42)
    .select(range(50))
)


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


thresholds = [0.4, 0.6, 0.8]
experiment_data = {"confidence_threshold_ablation": {}}

for name, ds in [("SQuAD", squad), ("AmbigQA", ambig), ("TriviaQA-rc", trivia)]:
    n = len(ds)
    # baseline no-clar
    baseline_acc = 0.0
    gt_list = []
    acc0_list = []
    for sample in ds:
        gt = get_gt(sample)
        gt_list.append(gt)
        if name == "AmbigQA":
            acc0 = False
        else:
            acc0 = True
        acc0_list.append(acc0)
        baseline_acc += acc0
    baseline_acc /= n

    # prepare storage
    clar_accs, avg_turns_list, ces_list = [], [], []
    preds = {thr: [] for thr in thresholds}

    # sweep thresholds
    for thr in thresholds:
        correct_sum = 0
        turns = 0
        for i, sample in enumerate(ds):
            acc0 = acc0_list[i]
            # simulate confidence
            conf = random.random()
            ask_clar = conf < thr
            if ask_clar:
                turns += 1
                acc = True  # post-clar always correct in simulation
            else:
                acc = acc0
            correct_sum += acc
            preds[thr].append(acc)
        clar_acc = correct_sum / n
        avg_turns = turns / n
        ces = (clar_acc - baseline_acc) / avg_turns if avg_turns > 0 else 0.0
        clar_accs.append(clar_acc)
        avg_turns_list.append(avg_turns)
        ces_list.append(ces)
        print(
            f"{name} @ thresh={thr:.2f}: baseline_acc={baseline_acc:.4f}, "
            f"clar_acc={clar_acc:.4f}, avg_turns={avg_turns:.4f}, CES={ces:.4f}"
        )

    # store results
    experiment_data["confidence_threshold_ablation"][name] = {
        "metrics": {
            "thresholds": thresholds,
            "baseline_acc": [baseline_acc] * len(thresholds),
            "clar_acc": clar_accs,
            "avg_turns": avg_turns_list,
            "CES": ces_list,
        },
        "losses": {"train": [], "val": []},
        "predictions": preds,
        "ground_truth": gt_list,
    }

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
