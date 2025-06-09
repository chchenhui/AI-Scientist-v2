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

# Print test accuracy for each ablation and setting
for ablation, settings in experiment_data.items():
    for key, vals in settings.items():
        preds = vals["predictions"]
        gt = vals["ground_truth"]
        acc = np.mean(preds == gt)
        print(f"Ablation: {ablation}, {key}, Test Accuracy: {acc:.3f}")

# Plot training/validation accuracy curves per ablation
for ablation, settings in experiment_data.items():
    try:
        plt.figure()
        for key, vals in settings.items():
            tr = vals["metrics"]["train"]
            va = vals["metrics"]["val"]
            plt.plot(tr, label=f"{key} Train")
            plt.plot(va, "--", label=f"{key} Val")
        plt.title(
            f"{ablation.capitalize()} Ablation - Accuracy Curves\nDataset: Synthetic Logistic"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, f"{ablation}_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {ablation} accuracy plot: {e}")
        plt.close()

# Plot training/validation loss curves per ablation
for ablation, settings in experiment_data.items():
    try:
        plt.figure()
        for key, vals in settings.items():
            trl = vals["losses"]["train"]
            val = vals["losses"]["val"]
            plt.plot(trl, label=f"{key} Train")
            plt.plot(val, "--", label=f"{key} Val")
        plt.title(
            f"{ablation.capitalize()} Ablation - Loss Curves\nDataset: Synthetic Logistic"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, f"{ablation}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {ablation} loss plot: {e}")
        plt.close()
