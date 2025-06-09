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

focal = experiment_data.get("focal_loss", {})
for key, stats in focal.items():
    gamma = key.split("_")[1]
    train_losses = stats["losses"]["train"]
    val_losses = stats["losses"]["val"]
    orig_acc = stats["metrics"]["orig_acc"]
    aug_acc = stats["metrics"]["aug_acc"]
    epochs = np.arange(1, len(train_losses) + 1)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"MNIST Focal Loss Results (gamma={gamma})")
        axes[0].plot(epochs, train_losses, label="Train Loss")
        axes[0].plot(epochs, val_losses, label="Val Loss")
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[1].plot(epochs, orig_acc, label="Orig Test Acc")
        axes[1].plot(epochs, aug_acc, label="Aug Test Acc")
        axes[1].set_title("Accuracy Curves")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        filename = os.path.join(working_dir, f"mnist_focal_gamma_{gamma}_results.png")
        plt.savefig(filename)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for gamma={gamma}: {e}")
        plt.close("all")
