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

# Extract ablation results
data_root = experiment_data.get("multi_dataset_synthetic_ablation", {})

# Print final validation accuracies
for name, ds in data_root.items():
    for E, d in sorted(ds["EPOCHS"].items()):
        val_acc = d["metrics"]["val"][-1]
        print(f"Dataset={name}, EPOCHS={E}, final val_acc={val_acc:.4f}")

# Plot for each dataset
for name, ds in data_root.items():
    try:
        per_epoch = ds["EPOCHS"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Left subplot: Accuracy
        for E, d in sorted(per_epoch.items()):
            axes[0].plot(
                range(1, len(d["metrics"]["train"]) + 1),
                d["metrics"]["train"],
                label=f"Train E={E}",
            )
            axes[0].plot(
                range(1, len(d["metrics"]["val"]) + 1),
                d["metrics"]["val"],
                linestyle="--",
                label=f"Val E={E}",
            )
        axes[0].set_title("Left: Training & Validation Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        # Right subplot: Loss
        for E, d in sorted(per_epoch.items()):
            axes[1].plot(
                range(1, len(d["losses"]["train"]) + 1),
                d["losses"]["train"],
                label=f"Train E={E}",
            )
            axes[1].plot(
                range(1, len(d["losses"]["val"]) + 1),
                d["losses"]["val"],
                linestyle="--",
                label=f"Val E={E}",
            )
        axes[1].set_title("Right: Training & Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        fig.suptitle(f"Dataset: {name}")
        save_path = os.path.join(working_dir, f"{name}_train_val_acc_loss.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for dataset {name}: {e}")
        plt.close()  # Ensure figure is closed even on error
