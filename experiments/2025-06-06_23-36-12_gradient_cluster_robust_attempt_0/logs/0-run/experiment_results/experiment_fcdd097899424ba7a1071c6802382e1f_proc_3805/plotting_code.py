import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot weighted accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for norm, data in experiment_data.get("max_grad_norm", {}).items():
        rec = data["synthetic"]
        epochs = range(len(rec["metrics"]["train"]))
        axes[0].plot(epochs, rec["metrics"]["train"], label=f"norm={norm}")
        axes[1].plot(epochs, rec["metrics"]["val"], label=f"norm={norm}")
    axes[0].set_title("Training Weighted Accuracy")
    axes[1].set_title("Validation Weighted Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted Accuracy")
        ax.legend()
    fig.suptitle(
        "Synthetic Dataset Weighted Accuracy across Epochs\nLeft: Training, Right: Validation"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_weighted_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for norm, data in experiment_data.get("max_grad_norm", {}).items():
        rec = data["synthetic"]
        epochs = range(len(rec["losses"]["train"]))
        axes[0].plot(epochs, rec["losses"]["train"], label=f"norm={norm}")
        axes[1].plot(epochs, rec["losses"]["val"], label=f"norm={norm}")
    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle(
        "Synthetic Dataset Loss across Epochs\nLeft: Training, Right: Validation"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()
