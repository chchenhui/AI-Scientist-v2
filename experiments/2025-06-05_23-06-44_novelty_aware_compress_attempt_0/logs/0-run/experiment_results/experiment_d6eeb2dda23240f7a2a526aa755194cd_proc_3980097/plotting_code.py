import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load data and print final validation metrics
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    for mode, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            val_losses = data["losses"]["val"]
            val_ret = data["metrics"]["Memory Retention Ratio"]["val"]
            val_eme = data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"]
            if val_losses:
                print(
                    f"[{mode}][{ds_name}] Final Val Loss={val_losses[-1]:.4f}, "
                    f"Retention={val_ret[-1]:.4f}, Efficiency={val_eme[-1]:.4f}"
                )
except Exception as e:
    print(f"Error loading or printing metrics: {e}")

# Plot loss curves
try:
    plt.figure()
    plt.subplot(1, 2, 1)
    for mode, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            tr = data["losses"]["train"]
            plt.plot(tr, label=f"{ds_name}-{mode}")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    for mode, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            vl = data["losses"]["val"]
            plt.plot(vl, label=f"{ds_name}-{mode}")
    plt.title("Validation Loss")
    plt.suptitle("Loss Curves Across Epochs")
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot memory retention ratio
try:
    plt.figure()
    plt.subplot(1, 2, 1)
    for mode, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            tr = data["metrics"]["Memory Retention Ratio"]["train"]
            plt.plot(tr, label=f"{ds_name}-{mode}")
    plt.title("Train Retention Ratio")
    plt.legend()
    plt.subplot(1, 2, 2)
    for mode, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            vl = data["metrics"]["Memory Retention Ratio"]["val"]
            plt.plot(vl, label=f"{ds_name}-{mode}")
    plt.title("Val Retention Ratio")
    plt.suptitle("Memory Retention Ratio Across Epochs")
    plt.savefig(os.path.join(working_dir, "memory_retention_ratio.png"))
    plt.close()
except Exception as e:
    print(f"Error creating retention plot: {e}")
    plt.close()

# Plot entropy-weighted memory efficiency
try:
    plt.figure()
    plt.subplot(1, 2, 1)
    for mode, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            tr = data["metrics"]["Entropy-Weighted Memory Efficiency"]["train"]
            plt.plot(tr, label=f"{ds_name}-{mode}")
    plt.title("Train EME")
    plt.legend()
    plt.subplot(1, 2, 2)
    for mode, ds_dict in experiment_data.items():
        for ds_name, data in ds_dict.items():
            vl = data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"]
            plt.plot(vl, label=f"{ds_name}-{mode}")
    plt.title("Val EME")
    plt.suptitle("Entropy-Weighted Memory Efficiency")
    plt.savefig(os.path.join(working_dir, "entropy_weighted_efficiency.png"))
    plt.close()
except Exception as e:
    print(f"Error creating EME plot: {e}")
    plt.close()
