import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")

# Load experiment data
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Print final metrics
for ds, ds_data in experiment_data.items():
    try:
        final_val_loss = ds_data["losses"]["val"][-1]
        final_ret = ds_data["metrics"]["Memory Retention Ratio"]["val"][-1]
        final_eme = ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"][-1]
        print(
            f"Dataset {ds} | Final Val Loss: {final_val_loss:.4f}, "
            f"Retention: {final_ret:.4f}, Efficiency: {final_eme:.4f}"
        )
    except Exception as e:
        print(f"Error retrieving final metrics for {ds}: {e}")

# Loss curves comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        losses = ds_data["losses"]
        epochs = range(len(losses["train"]))
        plt.plot(epochs, losses["train"], label=f"{ds} train")
        plt.plot(epochs, losses["val"], linestyle="--", label=f"{ds} val")
    plt.title("Loss Curves Across Datasets\nComparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Memory retention ratio comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        mets = ds_data["metrics"]["Memory Retention Ratio"]
        epochs = range(len(mets["train"]))
        plt.plot(epochs, mets["train"], label=f"{ds} train")
        plt.plot(epochs, mets["val"], linestyle="--", label=f"{ds} val")
    plt.title("Memory Retention Ratios Across Datasets\nComparison")
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "retention_ratio_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating retention ratio plot: {e}")
    plt.close()

# Entropy-weighted memory efficiency comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        mets = ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]
        epochs = range(len(mets["train"]))
        plt.plot(epochs, mets["train"], label=f"{ds} train")
        plt.plot(epochs, mets["val"], linestyle="--", label=f"{ds} val")
    plt.title("Entropy-Weighted Memory Efficiency Across Datasets\nComparison")
    plt.xlabel("Epoch")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "memory_efficiency_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating memory efficiency plot: {e}")
    plt.close()
