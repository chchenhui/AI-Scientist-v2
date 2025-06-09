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

# Plot WG accuracy and loss for each dataset
for name, data in experiment_data.items():
    metrics = data.get("metrics", {})
    losses = data.get("losses", {})

    # Worst‐group accuracy curve
    try:
        epochs = np.arange(len(metrics.get("train", [])))
        plt.figure()
        plt.plot(epochs, metrics["train"], label="Train WG Acc")
        plt.plot(epochs, metrics["val"], label="Val WG Acc")
        plt.title(
            f"{name} Dataset - Worst‐Group Accuracy\nTrain (blue) vs Validation (orange)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Worst‐Group Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_wg_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {name} worst‐group accuracy plot: {e}")
        plt.close()

    # Loss curve
    try:
        epochs = np.arange(len(losses.get("train", [])))
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.title(f"{name} Dataset - Loss Curve\nTrain (blue) vs Validation (orange)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {name} loss curve plot: {e}")
        plt.close()
