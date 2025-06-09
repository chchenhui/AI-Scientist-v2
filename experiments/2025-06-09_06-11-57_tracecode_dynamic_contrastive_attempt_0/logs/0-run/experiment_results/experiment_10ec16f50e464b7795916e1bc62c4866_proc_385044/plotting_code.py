import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}
exp = data.get("hard_negative_pool_size", {}).get("synthetic", {})
pool_sizes = exp.get("pool_sizes", [])
loss_train = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
metrics_train = exp.get("metrics", {}).get("train", [])
metrics_val = exp.get("metrics", {}).get("val", [])
epochs = list(range(len(loss_train[0]))) if loss_train else []

# Plot loss curves
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ps, lt in zip(pool_sizes, loss_train):
        axs[0].plot(epochs, lt, label=f"pool {ps}")
    axs[0].set_title("Left: Training Loss (synthetic)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    for ps, lv in zip(pool_sizes, loss_val):
        axs[1].plot(epochs, lv, label=f"pool {ps}")
    axs[1].set_title("Right: Validation Loss (synthetic)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    fig.suptitle("Loss Curves - Synthetic dataset")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ps, mt in zip(pool_sizes, metrics_train):
        axs[0].plot(epochs, mt, label=f"pool {ps}")
    axs[0].set_title("Left: Training Accuracy (synthetic)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    for ps, mv in zip(pool_sizes, metrics_val):
        axs[1].plot(epochs, mv, label=f"pool {ps}")
    axs[1].set_title("Right: Validation Accuracy (synthetic)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    fig.suptitle("Retrieval Accuracy Curves - Synthetic dataset")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "synthetic_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()
