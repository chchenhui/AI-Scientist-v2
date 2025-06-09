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

# Known learning rates
lrs = [1e-4, 1e-3, 1e-2]

# Iterate datasets
for ds_name, ds_data in experiment_data.get("multiple_synthetic", {}).items():
    metrics_train = ds_data["metrics"]["train"]  # shape (n_lrs, n_epochs)
    metrics_val = ds_data["metrics"]["val"]
    losses_train = ds_data["losses"]["train"]
    losses_val = ds_data["losses"]["val"]
    sp_corr = ds_data.get("spurious_corr")
    dim = ds_data.get("dim")
    n_epochs = metrics_train.shape[1]

    # Plot weighted accuracy curves
    try:
        plt.figure()
        epochs = np.arange(n_epochs)
        for i, lr in enumerate(lrs):
            plt.plot(epochs, metrics_train[i], "--", label=f"Train lr={lr}")
            plt.plot(epochs, metrics_val[i], "-", label=f"Val lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title(f"{ds_name} (corr={sp_corr}, dim={dim}): Weighted Accuracy")
        plt.legend()
        out_file = os.path.join(working_dir, f"{ds_name}_accuracy_curve.png")
        plt.savefig(out_file)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # Plot loss curves
    try:
        plt.figure()
        epochs = np.arange(n_epochs)
        for i, lr in enumerate(lrs):
            plt.plot(epochs, losses_train[i], "--", label=f"Train lr={lr}")
            plt.plot(epochs, losses_val[i], "-", label=f"Val lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} (corr={sp_corr}, dim={dim}): Loss Curve")
        plt.legend()
        out_file = os.path.join(working_dir, f"{ds_name}_loss_curve.png")
        plt.savefig(out_file)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()
