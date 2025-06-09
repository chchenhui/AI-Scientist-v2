import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for ablation, ab_data in experiment_data.items():
    try:
        plt.figure()
        for ds_key, ds_data in ab_data.items():
            epochs = range(1, len(ds_data["losses"]["train"]) + 1)
            plt.plot(epochs, ds_data["losses"]["train"], label=f"{ds_key} train")
            plt.plot(epochs, ds_data["losses"]["val"], label=f"{ds_key} val")
        plt.title(f"{ablation} Loss Curves\nDatasets comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ablation}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ablation}: {e}")
        plt.close()

    try:
        plt.figure()
        for ds_key, ds_data in ab_data.items():
            epochs = range(
                1, len(ds_data["metrics"]["Entropy Retention Ratio"]["train"]) + 1
            )
            plt.plot(
                epochs,
                ds_data["metrics"]["Entropy Retention Ratio"]["train"],
                label=f"{ds_key} train",
            )
            plt.plot(
                epochs,
                ds_data["metrics"]["Entropy Retention Ratio"]["val"],
                label=f"{ds_key} val",
            )
        plt.title(f"{ablation} Entropy Retention Ratio\nDatasets comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Entropy Retention Ratio")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{ablation}_entropy_retention_ratio_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating entropy retention ratio plot for {ablation}: {e}")
        plt.close()

    try:
        plt.figure()
        for ds_key, ds_data in ab_data.items():
            epochs = range(
                1,
                len(ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]["train"])
                + 1,
            )
            plt.plot(
                epochs,
                ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]["train"],
                label=f"{ds_key} train",
            )
            plt.plot(
                epochs,
                ds_data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"],
                label=f"{ds_key} val",
            )
        plt.title(f"{ablation} Entropy-Weighted Memory Efficiency\nDatasets comparison")
        plt.xlabel("Epoch")
        plt.ylabel("Entropy-Weighted Memory Efficiency")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"{ablation}_entropy_weighted_memory_efficiency_curves.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating memory efficiency plot for {ablation}: {e}")
        plt.close()
