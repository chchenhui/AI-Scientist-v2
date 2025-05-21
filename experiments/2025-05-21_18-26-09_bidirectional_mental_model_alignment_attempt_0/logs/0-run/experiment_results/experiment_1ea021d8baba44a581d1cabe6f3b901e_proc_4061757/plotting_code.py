import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# Dataset-specific loss and alignment plots
for name, d in data.items():
    try:
        epochs = range(1, len(d["losses"]["train"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, d["losses"]["train"], label="train")
        axes[0].plot(epochs, d["losses"]["val"], label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(epochs, d["alignments"]["train"], label="train")
        axes[1].plot(epochs, d["alignments"]["val"], label="val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Alignment (1-JSD)")
        axes[1].set_title("Alignment")
        axes[1].legend()
        fig.suptitle(f"{name} Metrics\nLeft: Loss, Right: Alignment (1-JSD)")
        plt.savefig(os.path.join(working_dir, f"{name}_loss_alignment.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {name} loss_alignment plot: {e}")
        plt.close()

# Dataset-specific MAI curves
for name, d in data.items():
    try:
        epochs = range(1, len(d["mai"]) + 1)
        plt.figure()
        plt.plot(epochs, d["mai"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("MAI")
        plt.title(f"{name} MAI over Epochs")
        plt.savefig(os.path.join(working_dir, f"{name}_mai_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {name} MAI plot: {e}")
        plt.close()

# Comparison of validation loss across datasets
try:
    plt.figure()
    for name, d in data.items():
        plt.plot(
            range(1, len(d["losses"]["val"]) + 1),
            d["losses"]["val"],
            marker="o",
            label=name,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Comparison of Validation Loss Across Datasets")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "comparison_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison validation loss plot: {e}")
    plt.close()

# Comparison of MAI across datasets
try:
    plt.figure()
    for name, d in data.items():
        plt.plot(range(1, len(d["mai"]) + 1), d["mai"], marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("MAI")
    plt.title("Comparison of MAI Across Datasets")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "comparison_mai.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison MAI plot: {e}")
    plt.close()
