import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# Iterate ratios
for ratio, data in exp.get("class_imbalance", {}).items():
    # Compute test accuracies
    accs = {}
    for key, d in data.items():
        preds = d["predictions"]
        gt = d["ground_truth"]
        accs[key] = np.mean(preds == gt)
    # Print metrics
    print(f"{ratio} test accuracies:")
    for k, v in accs.items():
        print(f"  {k}: {v:.3f}")
    # Bar plot of test accuracies
    try:
        plt.figure()
        keys, vals = zip(*sorted(accs.items()))
        plt.bar(range(len(vals)), vals, tick_label=keys)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title(f"Test Accuracy for Class Imbalance {ratio}")
        plt.xlabel("Batch Size Settings")
        plt.ylabel("Accuracy")
        fname = f"classimbalance_{ratio}_test_accuracy.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating plot_{ratio}_testacc: {e}")
    finally:
        plt.close()

# Training/validation curves for two representative settings
for ratio in list(exp.get("class_imbalance", {}))[:2]:
    data = exp["class_imbalance"][ratio]
    # pick representative key
    key = "ai_bs_32_user_bs_32"
    if key not in data:
        key = next(iter(data))
    tr = data[key]["metrics"]["train"]
    va = data[key]["metrics"]["val"]
    try:
        plt.figure()
        plt.plot(tr, label="Train Acc")
        plt.plot(va, label="Val Acc")
        plt.title(f"Train/Val Accuracy Curves {ratio}\n(ai_bs_32_user_bs_32)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"classimbalance_{ratio}_trainval_ai32_user32.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating plot_{ratio}_trainval: {e}")
    finally:
        plt.close()
