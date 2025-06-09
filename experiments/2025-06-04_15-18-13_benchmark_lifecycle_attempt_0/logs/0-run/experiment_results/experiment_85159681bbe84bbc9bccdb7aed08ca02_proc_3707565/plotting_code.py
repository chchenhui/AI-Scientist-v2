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

# Print final accuracies
for key, data in experiment_data.get("label_smoothing", {}).items():
    eps = key.split("_")[1]
    orig_acc = data["metrics"]["orig_acc"][-1]
    aug_acc = data["metrics"]["aug_acc"][-1]
    print(
        f"ε={eps}: original accuracy = {orig_acc:.4f}, augmented accuracy = {aug_acc:.4f}"
    )

# Plot loss curves
try:
    plt.figure()
    for key, data in experiment_data.get("label_smoothing", {}).items():
        eps = key.split("_")[1]
        train_losses = data["losses"]["train"]
        val_losses = data["losses"]["val"]
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f"ε={eps} train")
        plt.plot(epochs, val_losses, linestyle="--", label=f"ε={eps} val")
    plt.title("MNIST Loss Curves Across Label Smoothing Values")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mnist_loss_curves_label_smoothing.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure()
    for key, data in experiment_data.get("label_smoothing", {}).items():
        eps = key.split("_")[1]
        orig_accs = data["metrics"]["orig_acc"]
        aug_accs = data["metrics"]["aug_acc"]
        epochs = np.arange(1, len(orig_accs) + 1)
        plt.plot(epochs, orig_accs, label=f"ε={eps} orig_acc")
        plt.plot(epochs, aug_accs, linestyle="--", label=f"ε={eps} aug_acc")
    plt.title("MNIST Accuracy Curves Across Label Smoothing Values")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mnist_accuracy_curves_label_smoothing.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
