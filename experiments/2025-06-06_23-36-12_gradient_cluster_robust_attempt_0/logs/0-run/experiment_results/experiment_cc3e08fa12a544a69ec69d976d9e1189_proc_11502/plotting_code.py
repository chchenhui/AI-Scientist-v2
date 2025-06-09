import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    data = experiment_data["learning_rate"]["synthetic"]
    lrs = data["lrs"]
    metrics_train = data["metrics"]["train"]
    metrics_val = data["metrics"]["val"]
    losses_train = data["losses"]["train"]
    losses_val = data["losses"]["val"]
except Exception as e:
    print(f"Error loading or extracting experiment data: {e}")

# Plot worst‐group accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, metrics_train.shape[1] + 1)
    for i, lr in enumerate(lrs):
        axes[0].plot(epochs, metrics_train[i], label=f"lr={lr}")
        axes[1].plot(epochs, metrics_val[i], label=f"lr={lr}")
    axes[0].set_title("Training WG Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Worst‐group Accuracy")
    axes[1].set_title("Validation WG Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Worst‐group Accuracy")
    for ax in axes:
        ax.legend()
    fig.suptitle(
        "Synthetic dataset - Worst‐group Accuracy\nLeft: Training, Right: Validation"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_worst_group_accuracy.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close("all")

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, losses_train.shape[1] + 1)
    for i, lr in enumerate(lrs):
        axes[0].plot(epochs, losses_train[i], label=f"lr={lr}")
        axes[1].plot(epochs, losses_val[i], label=f"lr={lr}")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.legend()
    fig.suptitle("Synthetic dataset - Loss Curves\nLeft: Training, Right: Validation")
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close("all")
