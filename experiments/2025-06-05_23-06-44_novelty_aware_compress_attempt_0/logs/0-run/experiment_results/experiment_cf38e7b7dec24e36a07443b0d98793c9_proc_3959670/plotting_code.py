import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Print final evaluation metrics for each dataset
for ds, ds_data in experiment_data.items():
    try:
        ft = ds_data["losses"]["train"][-1]
        fv = ds_data["losses"]["val"][-1]
        et = ds_data["metrics"]["train"][-1]["entropy_eff"]
        ev = ds_data["metrics"]["val"][-1]["entropy_eff"]
        print(
            f"Dataset {ds} Final Train Loss: {ft:.4f}, Final Val Loss: {fv:.4f}, Final Train Eff: {et:.4f}, Final Val Eff: {ev:.4f}"
        )
    except Exception as e:
        print(f"Error printing final metrics for {ds}: {e}")

# Dataset-specific plots
for ds, ds_data in experiment_data.items():
    try:
        plt.figure()
        epochs = range(1, len(ds_data["losses"]["train"]) + 1)
        plt.plot(epochs, ds_data["losses"]["train"], label="Train")
        plt.plot(epochs, ds_data["losses"]["val"], linestyle="--", label="Val")
        plt.title(f"{ds} Loss Curves\nDataset: {ds}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()
    try:
        plt.figure()
        te = [m["entropy_eff"] for m in ds_data["metrics"]["train"]]
        ve = [m["entropy_eff"] for m in ds_data["metrics"]["val"]]
        epochs_e = range(1, len(te) + 1)
        plt.plot(epochs_e, te, label="Train")
        plt.plot(epochs_e, ve, linestyle="--", label="Val")
        plt.title(f"{ds} Entropy Efficiency Curves\nDataset: {ds}")
        plt.xlabel("Epoch")
        plt.ylabel("Entropy Efficiency")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_entropy_efficiency.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating entropy efficiency plot for {ds}: {e}")
        plt.close()

# Combined validation loss comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        epochs = range(1, len(ds_data["losses"]["val"]) + 1)
        plt.plot(epochs, ds_data["losses"]["val"], label=ds)
    plt.title("Validation Loss Comparison Across Datasets\nAll Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_val_loss_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating combined val loss plot: {e}")
    plt.close()

# Combined entropy efficiency comparison
try:
    plt.figure()
    for ds, ds_data in experiment_data.items():
        ve = [m["entropy_eff"] for m in ds_data["metrics"]["val"]]
        epochs_e = range(1, len(ve) + 1)
        plt.plot(epochs_e, ve, label=ds)
    plt.title("Validation Entropy Efficiency Comparison Across Datasets\nAll Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy Efficiency")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "all_datasets_entropy_efficiency_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating combined entropy efficiency plot: {e}")
    plt.close()
