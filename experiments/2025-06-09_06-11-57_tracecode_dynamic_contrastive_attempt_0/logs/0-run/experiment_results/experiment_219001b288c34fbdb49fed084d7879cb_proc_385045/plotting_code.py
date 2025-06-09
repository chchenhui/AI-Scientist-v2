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

# Loss curves
try:
    data = experiment_data["triplet_margin_sweep"]["synthetic"]
    margins = data["margins"]
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    epochs = range(1, len(train_losses[0]) + 1)
    plt.figure()
    for i, m in enumerate(margins):
        plt.plot(epochs, train_losses[i], label=f"Train m={m}")
        plt.plot(epochs, val_losses[i], "--", label=f"Val m={m}")
    plt.title("Triplet Margin Sweep on Synthetic Dataset - Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "triplet_margin_sweep_synthetic_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Retrieval accuracy curves
try:
    train_acc = data["metrics"]["train"]
    val_acc = data["metrics"]["val"]
    plt.figure()
    for i, m in enumerate(margins):
        plt.plot(epochs, train_acc[i], label=f"Train m={m}")
        plt.plot(epochs, val_acc[i], "--", label=f"Val m={m}")
    plt.title("Triplet Margin Sweep on Synthetic Dataset - Retrieval Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "triplet_margin_sweep_synthetic_accuracy.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
