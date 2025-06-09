import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    datasets = list(experiment_data.keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Loss curves comparison
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ds in datasets:
        metrics = experiment_data[ds]["metrics"]
        epochs = np.arange(1, len(metrics["train_loss"]) + 1)
        axes[0].plot(epochs, metrics["train_loss"], label=ds)
        axes[1].plot(epochs, metrics["val_loss"], label=ds)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    for ax in axes:
        ax.legend()
    fig.suptitle("Loss Curves Comparison\nLeft: Training, Right: Validation")
    plt.savefig(os.path.join(working_dir, "loss_curves_comparison.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss comparison plot: {e}")
    plt.close("all")

# Worst‐group accuracy comparison
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ds in datasets:
        metrics = experiment_data[ds]["metrics"]
        epochs = np.arange(1, len(metrics["train_wg"]) + 1)
        axes[0].plot(epochs, metrics["train_wg"], label=ds)
        axes[1].plot(epochs, metrics["val_wg"], label=ds)
    axes[0].set_title("Training Worst‐group Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Validation Worst‐group Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.legend()
    fig.suptitle("Worst‐group Accuracy Comparison\nLeft: Training, Right: Validation")
    plt.savefig(os.path.join(working_dir, "wg_accuracy_comparison.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating wg accuracy comparison plot: {e}")
    plt.close("all")

# NMI across epochs comparison
try:
    plt.figure(figsize=(6, 4))
    for ds in datasets:
        nmi = experiment_data[ds]["metrics"]["nmi"]
        epochs = np.arange(1, len(nmi) + 1)
        plt.plot(epochs, nmi, label=ds)
    plt.title("NMI across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("NMI")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "nmi_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating NMI comparison plot: {e}")
    plt.close("all")
