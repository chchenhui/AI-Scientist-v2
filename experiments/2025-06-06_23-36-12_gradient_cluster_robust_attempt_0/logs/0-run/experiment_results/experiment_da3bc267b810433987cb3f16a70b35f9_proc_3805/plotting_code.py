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

sd = experiment_data["batch_size"]["synthetic"]
batch_sizes = sd["batch_sizes"]
train_losses = sd["losses"]["train"]
val_losses = sd["losses"]["val"]
train_acc = sd["metrics"]["train"]
val_acc = sd["metrics"]["val"]

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, bs in enumerate(batch_sizes):
        axes[0].plot(train_losses[i], label=f"BS {bs}")
        axes[1].plot(val_losses[i], label=f"BS {bs}")
    axes[0].set_title("Train Loss")
    axes[1].set_title("Val Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("Synthetic Dataset Loss Curves\nLeft: Train Loss, Right: Val Loss")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot weighted group accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, bs in enumerate(batch_sizes):
        axes[0].plot(train_acc[i], label=f"BS {bs}")
        axes[1].plot(val_acc[i], label=f"BS {bs}")
    axes[0].set_title("Train Weighted Group Acc")
    axes[1].set_title("Val Weighted Group Acc")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted Group Accuracy")
        ax.legend()
    fig.suptitle("Synthetic Dataset Weighted Group Accuracy\nLeft: Train, Right: Val")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating wg accuracy curves: {e}")
    plt.close()
