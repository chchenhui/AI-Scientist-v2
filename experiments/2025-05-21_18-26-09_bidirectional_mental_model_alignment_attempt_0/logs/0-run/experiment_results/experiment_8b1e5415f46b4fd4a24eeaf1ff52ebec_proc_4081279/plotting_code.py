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

optimizers = list(experiment_data.keys())
datasets = list(experiment_data[optimizers[0]].keys())

# Plot loss curves
try:
    plt.figure(figsize=(12, 4))
    for i, ds in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), i + 1)
        for opt in optimizers:
            epochs = np.arange(1, len(experiment_data[opt][ds]["losses"]["train"]) + 1)
            ax.plot(
                epochs,
                experiment_data[opt][ds]["losses"]["train"],
                linestyle="-",
                label=f"{opt} Train",
            )
            ax.plot(
                epochs,
                experiment_data[opt][ds]["losses"]["val"],
                linestyle="--",
                label=f"{opt} Val",
            )
        ax.set_title(f"{ds} Dataset")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.suptitle("Loss Curves Across Datasets\nSolid: Train, Dashed: Val")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure(figsize=(12, 4))
    for i, ds in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), i + 1)
        for opt in optimizers:
            epochs = np.arange(
                1, len(experiment_data[opt][ds]["accuracy"]["train"]) + 1
            )
            ax.plot(
                epochs,
                experiment_data[opt][ds]["accuracy"]["train"],
                linestyle="-",
                label=f"{opt} Train",
            )
            ax.plot(
                epochs,
                experiment_data[opt][ds]["accuracy"]["val"],
                linestyle="--",
                label=f"{opt} Val",
            )
        ax.set_title(f"{ds} Dataset")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    plt.suptitle("Accuracy Curves Across Datasets\nSolid: Train, Dashed: Val")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# Plot alignment curves
try:
    plt.figure(figsize=(12, 4))
    for i, ds in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), i + 1)
        for opt in optimizers:
            epochs = np.arange(
                1, len(experiment_data[opt][ds]["alignments"]["train"]) + 1
            )
            ax.plot(
                epochs,
                experiment_data[opt][ds]["alignments"]["train"],
                linestyle="-",
                label=f"{opt} Train",
            )
            ax.plot(
                epochs,
                experiment_data[opt][ds]["alignments"]["val"],
                linestyle="--",
                label=f"{opt} Val",
            )
        ax.set_title(f"{ds} Dataset")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alignment")
        ax.legend()
    plt.suptitle("Alignment Curves Across Datasets\nSolid: Train, Dashed: Val")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment curves: {e}")
    plt.close()

# Plot MAI curves
try:
    plt.figure(figsize=(12, 4))
    for i, ds in enumerate(datasets):
        ax = plt.subplot(1, len(datasets), i + 1)
        for opt in optimizers:
            epochs = np.arange(1, len(experiment_data[opt][ds]["mai"]) + 1)
            ax.plot(epochs, experiment_data[opt][ds]["mai"], label=f"{opt}")
        ax.set_title(f"{ds} Dataset")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAI")
        ax.legend()
    plt.suptitle("MAI Across Datasets")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "mai_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MAI curves: {e}")
    plt.close()
