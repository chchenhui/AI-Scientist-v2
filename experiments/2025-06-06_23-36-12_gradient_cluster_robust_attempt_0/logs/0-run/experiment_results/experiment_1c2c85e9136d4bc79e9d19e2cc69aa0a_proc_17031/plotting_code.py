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

# Plot 1: Weighted Group Accuracy Curves
try:
    metrics = experiment_data["linear_classifier"]["synthetic"]["metrics"]
    train_acc = metrics["train"]
    val_acc = metrics["val"]
    epochs = train_acc.shape[1]
    lrs = [1e-4, 1e-3, 1e-2]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, lr in enumerate(lrs):
        axs[0].plot(range(1, epochs + 1), train_acc[i], marker="o", label=f"lr={lr}")
        axs[1].plot(range(1, epochs + 1), val_acc[i], marker="o", label=f"lr={lr}")
    axs[0].set_title("Training")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Weighted Group Accuracy")
    axs[0].legend()
    axs[1].set_title("Validation")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()
    fig.suptitle(
        "Weighted Group Accuracy Curves - Synthetic dataset\nLeft: Training, Right: Validation"
    )
    fig.savefig(os.path.join(working_dir, "synthetic_weighted_group_accuracy.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating weighted group accuracy plot: {e}")
    plt.close()

# Plot 2: Loss Curves
try:
    losses = experiment_data["linear_classifier"]["synthetic"]["losses"]
    train_loss = losses["train"]
    val_loss = losses["val"]
    epochs = train_loss.shape[1]
    lrs = [1e-4, 1e-3, 1e-2]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, lr in enumerate(lrs):
        axs[0].plot(range(1, epochs + 1), train_loss[i], marker="s", label=f"lr={lr}")
        axs[1].plot(range(1, epochs + 1), val_loss[i], marker="s", label=f"lr={lr}")
    axs[0].set_title("Training")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].set_title("Validation")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()
    fig.suptitle("Loss Curves - Synthetic dataset\nLeft: Training, Right: Validation")
    fig.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()
