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
    experiment_data = {}

lrs = [1e-4, 1e-3, 1e-2]
for dataset, info in experiment_data.get("NO_FEATURE_NORMALIZATION", {}).items():
    metrics_train = info["metrics"]["train"]
    metrics_val = info["metrics"]["val"]
    losses_train = info["losses"]["train"]
    losses_val = info["losses"]["val"]

    # Accuracy curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, lr in enumerate(lrs):
            epochs = np.arange(metrics_train.shape[1])
            axes[0].plot(epochs, metrics_train[i], label=f"lr={lr}")
            axes[1].plot(epochs, metrics_val[i], label=f"lr={lr}")
        fig.suptitle(f"{dataset} WG Accuracy Curves")
        axes[0].set_title("Left: Training WG Accuracy")
        axes[1].set_title("Right: Validation WG Accuracy")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("WG Accuracy")
            ax.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_accuracy.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating accuracy plot for {dataset}: {e}")
        plt.close()

    # Loss curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, lr in enumerate(lrs):
            epochs = np.arange(losses_train.shape[1])
            axes[0].plot(epochs, losses_train[i], label=f"lr={lr}")
            axes[1].plot(epochs, losses_val[i], label=f"lr={lr}")
        fig.suptitle(f"{dataset} Loss Curves")
        axes[0].set_title("Left: Training Loss")
        axes[1].set_title("Right: Validation Loss")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset}_loss.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating loss plot for {dataset}: {e}")
        plt.close()
