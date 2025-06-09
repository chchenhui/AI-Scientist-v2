import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Select five evenly spaced configurations
keys = list(exp["batch_size"].keys())
if len(keys) > 5:
    idxs = [int(i * (len(keys) - 1) / 4) for i in range(5)]
else:
    idxs = list(range(len(keys)))
selected = [keys[i] for i in idxs]

# Plot train/val accuracy and loss for each selected config
for key in selected:
    try:
        data = exp["batch_size"][key]
        epochs = range(1, len(data["metrics"]["train"]) + 1)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # Accuracy subplot
        axs[0].plot(epochs, data["metrics"]["train"], label="Train")
        axs[0].plot(epochs, data["metrics"]["val"], label="Validation")
        axs[0].set_title("Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].legend()
        # Loss subplot
        axs[1].plot(epochs, data["losses"]["train"], label="Train")
        axs[1].plot(epochs, data["losses"]["val"], label="Validation")
        axs[1].set_title("Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].legend()
        # Composite title
        fig.suptitle(f"Synthetic binary dataset - {key}")
        # Save and close
        fname = os.path.join(working_dir, f"{key}_train_val_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {key}: {e}")
        plt.close()
