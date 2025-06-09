import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot 1: Loss curves
try:
    plt.figure()
    for key, d in experiment_data["usage_based"].items():
        epochs = np.arange(1, len(d["losses"]["train"]) + 1)
        plt.plot(epochs, d["losses"]["train"], marker="o", label=f"{key} train")
        plt.plot(epochs, d["losses"]["val"], marker="x", label=f"{key} val")
    plt.title("Usage-based Model Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "usage_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: Retention ratio curves
try:
    plt.figure()
    for key, d in experiment_data["usage_based"].items():
        epochs = np.arange(1, len(d["metrics"]["train"]) + 1)
        plt.plot(
            epochs, d["metrics"]["train"], marker="o", label=f"{key} train retention"
        )
        plt.plot(epochs, d["metrics"]["val"], marker="x", label=f"{key} val retention")
    plt.title("Usage-based Model Retention Ratios")
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "usage_retention_ratios.png"))
    plt.close()
except Exception as e:
    print(f"Error creating retention ratios plot: {e}")
    plt.close()

# Plot 3: Predictions vs Ground Truth
try:
    keys = list(experiment_data["usage_based"].keys())
    n = len(keys)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    for i, key in enumerate(keys):
        d = experiment_data["usage_based"][key]
        gt = d["ground_truth"][:100]
        pred = d["predictions"][:100]
        axes[i, 0].plot(gt, color="blue")
        axes[i, 0].set_title(f"{key} Ground Truth")
        axes[i, 1].plot(pred, color="orange")
        axes[i, 1].set_title(f"{key} Predictions")
    fig.suptitle(
        "Usage-based Predictions vs Ground Truth (Left: Ground Truth, Right: Predictions)"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "usage_predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
