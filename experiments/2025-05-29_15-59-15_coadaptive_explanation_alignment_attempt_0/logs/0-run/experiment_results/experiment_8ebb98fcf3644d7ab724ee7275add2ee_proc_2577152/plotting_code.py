import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot accuracy curves
try:
    plt.figure()
    for key, entry in experiment_data.get("temperature_scaling", {}).items():
        train_acc = entry["metrics"]["train"]
        val_acc = entry["metrics"]["val"]
        epochs = np.arange(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, label=f"{key} train")
        plt.plot(epochs, val_acc, "--", label=f"{key} val")
    plt.title(
        "Temperature Scaling: Training and Validation Accuracy (Synthetic Binary Dataset)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "temperature_scaling_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot loss curves
try:
    plt.figure()
    for key, entry in experiment_data.get("temperature_scaling", {}).items():
        train_loss = entry["losses"]["train"]
        val_loss = entry["losses"]["val"]
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label=f"{key} train")
        plt.plot(epochs, val_loss, "--", label=f"{key} val")
    plt.title(
        "Temperature Scaling: Training and Validation Loss (Synthetic Binary Dataset)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "temperature_scaling_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Print test accuracies
for key, entry in experiment_data.get("temperature_scaling", {}).items():
    preds = entry["predictions"]
    gt = entry["ground_truth"]
    acc = (preds == gt).mean()
    print(f"{key}: Test accuracy = {acc:.4f}")
