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
    experiment_data = {}
wd_data = experiment_data.get("weight_decay", {})

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, data in wd_data.items():
        epochs = range(1, len(data["losses"]["train"]) + 1)
        axes[0].plot(epochs, data["losses"]["train"], label=key)
        axes[1].plot(epochs, data["losses"]["val"], label=key)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    fig.suptitle("MNIST Loss Curves\nLeft: Training Loss, Right: Validation Loss")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "mnist_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, data in wd_data.items():
        epochs = range(1, len(data["metrics"]["orig_acc"]) + 1)
        axes[0].plot(epochs, data["metrics"]["orig_acc"], label=key)
        axes[1].plot(epochs, data["metrics"]["aug_acc"], label=key)
    axes[0].set_title("Original Test Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Augmented Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle(
        "MNIST Accuracy Curves\nLeft: Original Test Accuracy, Right: Augmented Test Accuracy"
    )
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "mnist_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# Plot final accuracy vs weight decay
try:
    # Gather and sort by weight decay
    wds, origs, augs = [], [], []
    for key, data in wd_data.items():
        try:
            wd = float(key.split("_", 1)[1])
        except:
            wd = 0.0
        wds.append(wd)
        origs.append(data["metrics"]["orig_acc"][-1])
        augs.append(data["metrics"]["aug_acc"][-1])
    wds, origs, augs = zip(*sorted(zip(wds, origs, augs)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(wds, origs, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_title("Original Test Accuracy")
    axes[0].set_xlabel("Weight Decay")
    axes[0].set_ylabel("Accuracy")
    axes[1].plot(wds, augs, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_title("Augmented Test Accuracy")
    axes[1].set_xlabel("Weight Decay")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle(
        "MNIST Final Accuracy vs Weight Decay\nLeft: Original, Right: Augmented"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "mnist_final_accuracy_vs_wd.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy vs weight decay plot: {e}")
    plt.close()
