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

# Print final validation metrics
for ablation, ds_dict in experiment_data.items():
    for dataset_key, ed in ds_dict.items():
        val_loss = ed["losses"]["val"][-1]
        val_ratio = ed["metrics"]["Memory Retention Ratio"]["val"][-1]
        val_eme = ed["metrics"]["Entropy-Weighted Memory Efficiency"]["val"][-1]
        print(
            f"{ablation} | {dataset_key} - Final Val Loss: {val_loss:.4f}, "
            f"Retention Ratio: {val_ratio:.4f}, EME: {val_eme:.4f}"
        )

# Plotting curves for each dataset
dataset_keys = (
    list(experiment_data[next(iter(experiment_data))].keys()) if experiment_data else []
)
for dataset_key in dataset_keys:
    # Loss curve
    try:
        plt.figure()
        for ablation, ds_dict in experiment_data.items():
            ed = ds_dict[dataset_key]
            epochs = range(len(ed["losses"]["train"]))
            plt.plot(epochs, ed["losses"]["train"], label=f"{ablation} train")
            plt.plot(epochs, ed["losses"]["val"], "--", label=f"{ablation} val")
        plt.title(
            f"{dataset_key}: Training and Validation Loss\nSolid: Train, Dashed: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_key}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_key}: {e}")
        plt.close()

    # Memory Retention Ratio
    try:
        plt.figure()
        for ablation, ds_dict in experiment_data.items():
            ed = ds_dict[dataset_key]
            epochs = range(len(ed["metrics"]["Memory Retention Ratio"]["train"]))
            plt.plot(
                epochs,
                ed["metrics"]["Memory Retention Ratio"]["train"],
                label=f"{ablation} train",
            )
            plt.plot(
                epochs,
                ed["metrics"]["Memory Retention Ratio"]["val"],
                "--",
                label=f"{ablation} val",
            )
        plt.title(
            f"{dataset_key}: Memory Retention Ratio\nSolid: Train, Dashed: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Memory Retention Ratio")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{dataset_key}_memory_retention_ratio.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating retention ratio plot for {dataset_key}: {e}")
        plt.close()

    # Entropy-Weighted Memory Efficiency
    try:
        plt.figure()
        for ablation, ds_dict in experiment_data.items():
            ed = ds_dict[dataset_key]
            epochs = range(
                len(ed["metrics"]["Entropy-Weighted Memory Efficiency"]["train"])
            )
            plt.plot(
                epochs,
                ed["metrics"]["Entropy-Weighted Memory Efficiency"]["train"],
                label=f"{ablation} train",
            )
            plt.plot(
                epochs,
                ed["metrics"]["Entropy-Weighted Memory Efficiency"]["val"],
                "--",
                label=f"{ablation} val",
            )
        plt.title(
            f"{dataset_key}: Entropy-Weighted Memory Efficiency\nSolid: Train, Dashed: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Entropy-Weighted Memory Efficiency")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"{dataset_key}_entropy_weighted_memory_efficiency.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating EME plot for {dataset_key}: {e}")
        plt.close()
