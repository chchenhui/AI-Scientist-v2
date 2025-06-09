import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = list(experiment_data.get("original", {}).keys())
ablations = ["original", "recency"]
for ds in datasets:
    # prepare curves
    curves = {}
    for ab in ablations:
        ed = experiment_data[ab][ds]
        curves[ab] = {
            "loss_train": ed["losses"]["train"],
            "loss_val": ed["losses"]["val"],
            "mrr_train": ed["metrics"]["Memory Retention Ratio"]["train"],
            "mrr_val": ed["metrics"]["Memory Retention Ratio"]["val"],
            "ewme_train": ed["metrics"]["Entropy-Weighted Memory Efficiency"]["train"],
            "ewme_val": ed["metrics"]["Entropy-Weighted Memory Efficiency"]["val"],
        }
    epochs = range(1, len(curves["original"]["loss_train"]) + 1)
    # Plot Loss
    try:
        plt.figure()
        for ab in ablations:
            plt.plot(epochs, curves[ab]["loss_train"], label=f"{ab} train")
            plt.plot(epochs, curves[ab]["loss_val"], "--", label=f"{ab} val")
        plt.title(f"{ds} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {ds} loss plot: {e}")
        plt.close()
    # Plot Memory Retention Ratio
    try:
        plt.figure()
        for ab in ablations:
            plt.plot(epochs, curves[ab]["mrr_train"], label=f"{ab} train")
            plt.plot(epochs, curves[ab]["mrr_val"], "--", label=f"{ab} val")
        plt.title(f"{ds} Memory Retention Ratio")
        plt.xlabel("Epoch")
        plt.ylabel("Retention Ratio")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_mrr.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {ds} MRR plot: {e}")
        plt.close()
    # Plot Entropy-Weighted Memory Efficiency
    try:
        plt.figure()
        for ab in ablations:
            plt.plot(epochs, curves[ab]["ewme_train"], label=f"{ab} train")
            plt.plot(epochs, curves[ab]["ewme_val"], "--", label=f"{ab} val")
        plt.title(f"{ds} Entropy-Weighted Memory Efficiency")
        plt.xlabel("Epoch")
        plt.ylabel("Efficiency")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_ewme.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {ds} EWME plot: {e}")
        plt.close()
