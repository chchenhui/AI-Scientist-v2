import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_data = {}

# Known hyperparameter sweep
n_components_list = [20, 30, 40, 50]
metrics = exp_data.get("n_components", {}).get("synthetic", {}).get("metrics", {})

# Plot training error curves
try:
    plt.figure()
    for idx, n in enumerate(n_components_list):
        train_err = metrics.get("train", [])[idx]
        plt.plot(train_err, label=f"n_components={n}")
    plt.title("Training Error Curves - Synthetic Dataset\nn_components: 20,30,40,50")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_train_error.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training error plot: {e}")
    plt.close()

# Plot validation error curves
try:
    plt.figure()
    for idx, n in enumerate(n_components_list):
        val_err = metrics.get("val", [])[idx]
        plt.plot(val_err, label=f"n_components={n}")
    plt.title("Validation Error Curves - Synthetic Dataset\nn_components: 20,30,40,50")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_val_error.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation error plot: {e}")
    plt.close()
