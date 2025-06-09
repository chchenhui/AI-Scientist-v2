import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Prepare datasets and ablations
ablations = list(experiment_data.keys())
dataset_names = list(experiment_data.get("baseline", {}).keys())

# Plot metrics curves per dataset
for ds in dataset_names:
    try:
        fig = plt.figure(figsize=(10, 8))
        plt.suptitle(f"Training and Validation Metrics for {ds}")
        epochs = range(1, len(experiment_data["baseline"][ds]["losses"]["train"]) + 1)
        # Loss subplot
        ax1 = fig.add_subplot(3, 1, 1)
        for ab in ablations:
            lt = experiment_data[ab][ds]["losses"]["train"]
            lv = experiment_data[ab][ds]["losses"]["val"]
            ax1.plot(epochs, lt, label=f"{ab} train")
            ax1.plot(epochs, lv, linestyle="--", label=f"{ab} val")
        ax1.set_title("Loss vs Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        # Memory Retention Ratio subplot
        ax2 = fig.add_subplot(3, 1, 2)
        for ab in ablations:
            mrt = experiment_data[ab][ds]["metrics"]["Memory Retention Ratio"]["train"]
            mrv = experiment_data[ab][ds]["metrics"]["Memory Retention Ratio"]["val"]
            ax2.plot(epochs, mrt, label=f"{ab} train")
            ax2.plot(epochs, mrv, linestyle="--", label=f"{ab} val")
        ax2.set_title("Memory Retention Ratio vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Ratio")
        ax2.legend()
        # Entropy-Weighted Memory Efficiency subplot
        ax3 = fig.add_subplot(3, 1, 3)
        for ab in ablations:
            et = experiment_data[ab][ds]["metrics"][
                "Entropy-Weighted Memory Efficiency"
            ]["train"]
            ev = experiment_data[ab][ds]["metrics"][
                "Entropy-Weighted Memory Efficiency"
            ]["val"]
            ax3.plot(epochs, et, label=f"{ab} train")
            ax3.plot(epochs, ev, linestyle="--", label=f"{ab} val")
        ax3.set_title("Entropy-Weighted Memory Efficiency vs Epoch")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Efficiency")
        ax3.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"{ds}_metrics_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {ds}: {e}")
        plt.close()

# Plot sample comparisons for first two datasets only
for ds in dataset_names[:2]:
    try:
        gt_list = experiment_data["baseline"][ds]["ground_truth"]
        pred_list = experiment_data["baseline"][ds]["predictions"]
        if gt_list and pred_list:
            gt_seq = gt_list[0]
            pred_seq = pred_list[0]
            gt_str = "".join(chr(c) for c in gt_seq if c != 0)
            pred_str = "".join(chr(c) for c in pred_seq if c != 0)
        else:
            gt_str, pred_str = "", ""
        fig = plt.figure(figsize=(10, 4))
        plt.suptitle(f"{ds}: Left: Ground Truth, Right: Generated Samples")
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.text(0.5, 0.5, gt_str, ha="center", va="center", wrap=True)
        ax1.axis("off")
        ax1.set_title("Ground Truth")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.text(0.5, 0.5, pred_str, ha="center", va="center", wrap=True)
        ax2.axis("off")
        ax2.set_title("Generated Samples")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"{ds}_sample.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating sample plot for {ds}: {e}")
        plt.close()
