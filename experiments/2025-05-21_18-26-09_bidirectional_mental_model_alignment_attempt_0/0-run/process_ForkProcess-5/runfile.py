import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

sd = data.get("learning_rate", {}).get("synthetic", {})

# Plot 1: Loss curves
try:
    lrs = sd.get("lrs", [])
    train_losses = sd.get("losses", {}).get("train", [])
    val_losses = sd.get("losses", {}).get("val", [])
    epochs = range(1, len(train_losses[0]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lr, tr in zip(lrs, train_losses):
        axes[0].plot(epochs, tr, label=f"lr={lr}")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    for lr, vl in zip(lrs, val_losses):
        axes[1].plot(epochs, vl, label=f"lr={lr}")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Loss Curves\nLeft: Training Loss, Right: Validation Loss"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Alignment curves
try:
    train_align = sd.get("metrics", {}).get("train", [])
    val_align = sd.get("metrics", {}).get("val", [])
    epochs = range(1, len(train_align[0]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for lr, ta in zip(lrs, train_align):
        axes[0].plot(epochs, ta, label=f"lr={lr}")
    axes[0].set_title("Training Alignment")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Alignment (1-JSD)")
    axes[0].legend()
    for lr, va in zip(lrs, val_align):
        axes[1].plot(epochs, va, label=f"lr={lr}")
    axes[1].set_title("Validation Alignment")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Alignment (1-JSD)")
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Alignment Curves\nLeft: Training Alignment, Right: Validation Alignment"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()
