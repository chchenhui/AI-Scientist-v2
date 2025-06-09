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

# Plot loss and accuracy curves for each encoder
for name, ds in experiment_data.items():
    # Loss curves
    try:
        plt.figure()
        for E, data in ds["synthetic"].items():
            epochs = np.arange(1, len(data["losses"]["train"]) + 1)
            plt.plot(epochs, data["losses"]["train"], label=f"Train E={E}")
            plt.plot(epochs, data["losses"]["val"], "--", label=f"Val E={E}")
        plt.title(f"Synthetic dataset Loss Curves for {name.upper()} Encoder")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {name}: {e}")
        plt.close()

    # Accuracy curves
    try:
        plt.figure()
        for E, data in ds["synthetic"].items():
            epochs = np.arange(1, len(data["metrics"]["train"]) + 1)
            plt.plot(epochs, data["metrics"]["train"], label=f"Train Acc E={E}")
            plt.plot(epochs, data["metrics"]["val"], "--", label=f"Val Acc E={E}")
        plt.title(f"Synthetic dataset Retrieval Accuracy for {name.upper()} Encoder")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_synthetic_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {name}: {e}")
        plt.close()

# Print final validation accuracy for each setting
for name, ds in experiment_data.items():
    for E, data in ds["synthetic"].items():
        final_acc = data["metrics"]["val"][-1]
        print(f"{name} encoder, E={E}: final validation accuracy = {final_acc:.4f}")
