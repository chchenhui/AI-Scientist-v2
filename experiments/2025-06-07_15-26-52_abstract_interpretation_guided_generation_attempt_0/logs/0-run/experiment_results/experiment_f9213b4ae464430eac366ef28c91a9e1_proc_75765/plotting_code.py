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

d = experiment_data.get("learning_rate", {}).get("synthetic", {})
params = d.get("params", [])
loss_train = np.array(d.get("losses", {}).get("train", []))
loss_val = np.array(d.get("losses", {}).get("val", []))
metrics_train = np.array(d.get("metrics", {}).get("train", []))
metrics_val = np.array(d.get("metrics", {}).get("val", []))
epochs = np.arange(1, loss_train.shape[1] + 1) if loss_train.ndim == 2 else np.array([])

# Plot Loss Curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, lr in enumerate(params):
        axes[0].plot(epochs, loss_train[i], label=f"{lr}")
        axes[1].plot(epochs, loss_val[i], label=f"{lr}")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[0].legend(title="LR")
    axes[1].legend(title="LR")
    fig.suptitle(
        "Synthetic dataset Loss Curves\nLeft: Training Loss, Right: Validation Loss"
    )
    fig.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot AICR Curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, lr in enumerate(params):
        axes[0].plot(epochs, metrics_train[i], label=f"{lr}")
        axes[1].plot(epochs, metrics_val[i], label=f"{lr}")
    axes[0].set_title("Training AICR")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AICR")
    axes[1].set_title("Validation AICR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AICR")
    axes[0].legend(title="LR")
    axes[1].legend(title="LR")
    fig.suptitle(
        "Synthetic dataset AICR Curves\nLeft: Training AICR, Right: Validation AICR"
    )
    fig.savefig(os.path.join(working_dir, "synthetic_AICR_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating AICR curves plot: {e}")
    plt.close()
