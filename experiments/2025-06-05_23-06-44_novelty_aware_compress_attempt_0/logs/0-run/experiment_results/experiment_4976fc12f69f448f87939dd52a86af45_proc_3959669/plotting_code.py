import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")

# Load data
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Print final metrics
for ds, ds_data in experiment_data.items():
    try:
        final_val_loss = ds_data["losses"]["val"][-1]
        final_val_ratio = ds_data["metrics"]["Memory Retention Ratio"]["val"][-1]
        final_val_eme = ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"][
            -1
        ]
        print(
            f"{ds} Final Val Loss: {final_val_loss:.4f}, "
            f"Retention: {final_val_ratio:.4f}, EME: {final_val_eme:.4f}"
        )
    except Exception as e:
        print(f"Error printing final metrics for {ds}: {e}")

# Loss curves comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        epochs = range(len(ds_data["losses"]["train"]))
        plt.plot(epochs, ds_data["losses"]["train"], label=f"{ds} train")
        plt.plot(epochs, ds_data["losses"]["val"], linestyle="--", label=f"{ds} val")
    plt.title(
        "Loss Curves Across Datasets\nDatasets: " + ", ".join(experiment_data.keys())
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Memory retention ratio comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        m = ds_data["metrics"]["Memory Retention Ratio"]
        epochs = range(len(m["train"]))
        plt.plot(epochs, m["train"], label=f"{ds} train")
        plt.plot(epochs, m["val"], linestyle="--", label=f"{ds} val")
    plt.title(
        "Memory Retention Ratios Across Datasets\nDatasets: "
        + ", ".join(experiment_data.keys())
    )
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_retention_ratios.png"))
    plt.close()
except Exception as e:
    print(f"Error creating retention ratios plot: {e}")
    plt.close()

# Entropy-weighted memory efficiency comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        m = ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]
        epochs = range(len(m["train"]))
        plt.plot(epochs, m["train"], label=f"{ds} train")
        plt.plot(epochs, m["val"], linestyle="--", label=f"{ds} val")
    plt.title(
        "Entropy-Weighted Memory Efficiency Across Datasets\nDatasets: "
        + ", ".join(experiment_data.keys())
    )
    plt.xlabel("Epoch")
    plt.ylabel("Entropy-Weighted Memory Efficiency")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "all_datasets_entropy_weighted_memory_efficiency.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating efficiency plot: {e}")
    plt.close()
