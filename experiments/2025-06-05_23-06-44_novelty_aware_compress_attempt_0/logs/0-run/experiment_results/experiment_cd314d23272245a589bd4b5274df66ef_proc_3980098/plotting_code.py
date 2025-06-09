import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = {}
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

for key_type, metric_name, subtitle, filename in [
    ("losses", "Loss", "Loss over Epochs", "combined_loss_curve.png"),
    (
        "metrics",
        "Memory Retention Ratio",
        "Memory Retention Ratio over Epochs",
        "combined_memory_retention_ratio.png",
    ),
    (
        "metrics",
        "Entropy-Weighted Memory Efficiency",
        "Entropy-Weighted Memory Efficiency over Epochs",
        "combined_entropy_weighted_memory_efficiency.png",
    ),
]:
    try:
        plt.figure()
        for ablation, ds_dict in experiment_data.items():
            for dataset, data in ds_dict.items():
                if key_type == "losses":
                    train_vals = data["losses"]["train"]
                    val_vals = data["losses"]["val"]
                else:
                    train_vals = data["metrics"][metric_name]["train"]
                    val_vals = data["metrics"][metric_name]["val"]
                epochs = np.arange(1, len(train_vals) + 1)
                plt.plot(
                    epochs,
                    train_vals,
                    label=f"{ablation}_{dataset}_train",
                    linestyle="-",
                )
                plt.plot(
                    epochs, val_vals, label=f"{ablation}_{dataset}_val", linestyle="--"
                )
        plt.title(metric_name)
        plt.suptitle(subtitle)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.legend()
        plt.savefig(os.path.join(working_dir, filename))
    except Exception as e:
        print(f"Error creating plot for {metric_name}: {e}")
    finally:
        plt.close()
