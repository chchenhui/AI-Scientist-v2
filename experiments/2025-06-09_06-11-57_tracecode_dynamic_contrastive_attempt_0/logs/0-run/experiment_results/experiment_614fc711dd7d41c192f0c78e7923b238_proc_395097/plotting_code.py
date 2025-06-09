import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    synthetic_data = experiment_data["EPOCHS"]["synthetic"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    synthetic_data = {}

# Plot loss curves
try:
    plt.figure()
    for E, data in synthetic_data.items():
        epochs = list(range(1, len(data["losses"]["train"]) + 1))
        plt.plot(epochs, data["losses"]["train"], label=f"Train E={E}")
        plt.plot(epochs, data["losses"]["val"], linestyle="--", label=f"Val E={E}")
    plt.suptitle("Synthetic Dataset")
    plt.title("Train vs Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure()
    for E, data in synthetic_data.items():
        epochs = list(range(1, len(data["metrics"]["train"]) + 1))
        plt.plot(epochs, data["metrics"]["train"], label=f"Train E={E}")
        plt.plot(epochs, data["metrics"]["val"], linestyle="--", label=f"Val E={E}")
    plt.suptitle("Synthetic Dataset")
    plt.title("Train vs Validation Retrieval Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()
