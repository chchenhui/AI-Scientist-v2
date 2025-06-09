import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    synthetic = experiment_data["mini_batch_size"]["synthetic"]
    batch_sizes = synthetic["batch_sizes"]
    train_errs = synthetic["metrics"]["train"]
    val_errs = synthetic["metrics"]["val"]
    train_losses = synthetic["losses"]["train"]
    val_losses = synthetic["losses"]["val"]
    predictions = synthetic["predictions"]
    ground_truths = synthetic["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot 1: Error curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for errs, bs in zip(train_errs, batch_sizes):
        axes[0].plot(range(1, len(errs) + 1), errs, label=f"bs={bs}")
    for errs, bs in zip(val_errs, batch_sizes):
        axes[1].plot(range(1, len(errs) + 1), errs, label=f"bs={bs}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Error Curves\nLeft: Training Error, Right: Validation Error"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ls, bs in zip(train_losses, batch_sizes):
        axes[0].plot(range(1, len(ls) + 1), ls, label=f"bs={bs}")
    for ls, bs in zip(val_losses, batch_sizes):
        axes[1].plot(range(1, len(ls) + 1), ls, label=f"bs={bs}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset - Loss Curves\nLeft: Training Loss, Right: Validation Loss"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plots 3-6: Sample reconstructions for each batch size
for idx, bs in enumerate(batch_sizes, start=3):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        gt = ground_truths[idx - 3]
        pred = predictions[idx - 3]
        axes[0].plot(gt[0], color="blue")
        axes[1].plot(pred[0], color="orange")
        axes[0].set_title("Ground Truth Sample 0")
        axes[1].set_title("Predicted Sample 0")
        axes[0].set_xlabel("Dimension")
        axes[1].set_xlabel("Dimension")
        fig.suptitle(
            f"Synthetic Dataset - Sample Recon (bs={bs})\nLeft: Ground Truth, Right: Predicted"
        )
        plt.savefig(os.path.join(working_dir, f"synthetic_sample_bs_{bs}.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot{idx}: {e}")
        plt.close()
