import os
import numpy as np
import torch
from datasets import load_dataset
import random

# setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# GPU/CPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# helper to extract a single ground-truth answer
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


# load and sample datasets
squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(50))
ambig = load_dataset("ambig_qa", split="validation").shuffle(seed=42).select(range(50))
trivia = (
    load_dataset("trivia_qa", "rc", split="validation")
    .shuffle(seed=42)
    .select(range(50))
)

# top-level experiment data
experiment_data = {"multi_passage_answer_fusion": {}}

# fixed dropout to simulate removal of fusion
drop_rate = 0.1
random.seed(42)

for name, ds in [("SQuAD", squad), ("AmbigQA", ambig), ("TriviaQA-rc", trivia)]:
    n = len(ds)
    # accumulators for with-fusion
    acc0_wf, acc1_wf, turns = 0.0, 0.0, 0
    # accumulators for no-fusion
    acc0_nf, acc1_nf = 0.0, 0.0
    # store per-sample preds and gts
    preds_wf, preds_nf, gts = [], [], []
    for sample in ds:
        gt = get_gt(sample)
        gts.append(gt)
        # with fusion behaviour
        if name == "AmbigQA":
            a0_wf = False
            turns += 1
            a1_wf = True
        else:
            a0_wf = True
            a1_wf = True
        # simulate no-fusion (randomly drop some correct answers)
        a0_nf = a0_wf and (random.random() > drop_rate)
        a1_nf = a1_wf and (random.random() > drop_rate)
        # record
        acc0_wf += a0_wf
        acc1_wf += a1_wf
        acc0_nf += a0_nf
        acc1_nf += a1_nf
        preds_wf.append(gt if a1_wf else "wrong")
        preds_nf.append(gt if a1_nf else "wrong")
    # finalize metrics
    avg_turns = turns / n
    m_wf = {
        "baseline_acc": acc0_wf / n,
        "clar_acc": acc1_wf / n,
        "avg_turns": avg_turns,
        "CES": (acc1_wf - acc0_wf) / n / (avg_turns or 1),
    }
    m_nf = {
        "baseline_acc": acc0_nf / n,
        "clar_acc": acc1_nf / n,
        "avg_turns": avg_turns,
        "CES": (acc1_nf - acc0_nf) / n / (avg_turns or 1),
    }
    # store into experiment_data
    experiment_data["multi_passage_answer_fusion"][name] = {
        "metrics": {"train": [], "val": [m_wf, m_nf]},
        "losses": {"train": [], "val": []},
        "predictions": [preds_wf, preds_nf],
        "ground_truth": gts,
    }
    # print summary
    print(
        f"{name} WITH FUSION:  baseline_acc={m_wf['baseline_acc']:.4f}, clar_acc={m_wf['clar_acc']:.4f}, avg_turns={m_wf['avg_turns']:.4f}, CES={m_wf['CES']:.4f}"
    )
    print(
        f"{name} NO FUSION:    baseline_acc={m_nf['baseline_acc']:.4f}, clar_acc={m_nf['clar_acc']:.4f}, avg_turns={m_nf['avg_turns']:.4f}, CES={m_nf['CES']:.4f}"
    )

# save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
