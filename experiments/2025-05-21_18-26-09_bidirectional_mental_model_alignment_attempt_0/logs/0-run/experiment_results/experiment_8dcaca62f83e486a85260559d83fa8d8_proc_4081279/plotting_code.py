import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Determine datasets and ablations
ablations = list(experiment_data.keys())
datasets = list(next(iter(experiment_data.values()), {}).keys())

# Plot for each dataset
for dname in datasets:
    try:
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        # Epochs
        n_epochs = len(experiment_data[ablations[0]][dname]["losses"]["train"])
        epochs = np.arange(1, n_epochs + 1)
        # Plot each ablation
        for abl in ablations:
            losses = experiment_data[abl][dname]["losses"]
            metrics = experiment_data[abl][dname]["metrics"]
            ax1.plot(epochs, losses["train"], label=f"{abl} train")
            ax1.plot(epochs, losses["val"], "--", label=f"{abl} val")
            ax2.plot(epochs, metrics["train"], label=f"{abl} train")
            ax2.plot(epochs, metrics["val"], "--", label=f"{abl} val")
        # Titles and labels
        fig.suptitle(
            f"{dname} dataset\nLeft: Loss curves, Right: Bidirectional Alignment curves"
        )
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_title("Bidirectional Alignment")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Alignment")
        ax1.legend(fontsize="small")
        ax2.legend(fontsize="small")
        # Save and close
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_alignment.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dname}: {e}")
        plt.close()
