import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# Print final evaluation metrics for each dataset
for ds_name, ds_data in experiment_data.items():
    final_val_loss = ds_data["losses"]["val"][-1]
    final_val_eff = ds_data["metrics"]["val"][-1]
    print(
        f"Dataset {ds_name} Final Val Loss: {final_val_loss:.4f}, Final Entropy Efficiency: {final_val_eff:.4f}"
    )

# Per‐dataset plots
for ds_name, ds_data in experiment_data.items():
    epochs = range(len(ds_data["losses"]["train"]))
    # Loss curves
    try:
        plt.figure()
        plt.plot(epochs, ds_data["losses"]["train"], label="train")
        plt.plot(epochs, ds_data["losses"]["val"], linestyle="--", label="val")
        plt.title(f"{ds_name} Loss Curves Across Epochs\nDataset: {ds_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()
    # Retention ratios
    try:
        plt.figure()
        plt.plot(epochs, ds_data["metrics"]["train"], label="train")
        plt.plot(epochs, ds_data["metrics"]["val"], linestyle="--", label="val")
        plt.title(
            f"{ds_name} Memory Retention Ratios Across Epochs\nDataset: {ds_name}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Retention Ratio")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_retention_ratios.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating retention plot for {ds_name}: {e}")
        plt.close()

# Comparison plots across datasets
# Loss comparison
try:
    plt.figure()
    for ds_name, ds_data in experiment_data.items():
        epochs = range(len(ds_data["losses"]["train"]))
        plt.plot(epochs, ds_data["losses"]["train"], label=f"{ds_name} train")
        plt.plot(
            epochs, ds_data["losses"]["val"], linestyle="--", label=f"{ds_name} val"
        )
    plt.title("Loss Curves Across Datasets\nComparison of Train and Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "comparison_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison loss plot: {e}")
    plt.close()

# Retention comparison
try:
    plt.figure()
    for ds_name, ds_data in experiment_data.items():
        epochs = range(len(ds_data["metrics"]["train"]))
        plt.plot(epochs, ds_data["metrics"]["train"], label=f"{ds_name} train")
        plt.plot(
            epochs, ds_data["metrics"]["val"], linestyle="--", label=f"{ds_name} val"
        )
    plt.title(
        "Memory Retention Ratios Across Datasets\nComparison of Train and Validation"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "comparison_retention_ratios.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison retention plot: {e}")
    plt.close()
