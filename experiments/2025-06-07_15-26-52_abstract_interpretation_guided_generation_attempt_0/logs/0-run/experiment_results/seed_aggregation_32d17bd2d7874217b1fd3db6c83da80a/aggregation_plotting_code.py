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

# Extract data for synthetic dataset under varying learning rates
d = experiment_data.get("learning_rate", {}).get("synthetic", {})
loss_train = np.array(d.get("losses", {}).get("train", []))
loss_val = np.array(d.get("losses", {}).get("val", []))
metrics_train = np.array(d.get("metrics", {}).get("train", []))
metrics_val = np.array(d.get("metrics", {}).get("val", []))

# Ensure data is at least 2D so we can aggregate over runs
if loss_train.ndim == 1:
    loss_train = loss_train[np.newaxis, :]
    loss_val = loss_val[np.newaxis, :]
    metrics_train = metrics_train[np.newaxis, :]
    metrics_val = metrics_val[np.newaxis, :]

# Define epochs array
epochs = np.arange(1, loss_train.shape[1] + 1) if loss_train.ndim == 2 else np.array([])

# Compute mean and SEM for losses
mean_loss_train = np.mean(loss_train, axis=0)
sem_loss_train = np.std(loss_train, axis=0) / np.sqrt(loss_train.shape[0])
mean_loss_val = np.mean(loss_val, axis=0)
sem_loss_val = np.std(loss_val, axis=0) / np.sqrt(loss_val.shape[0])

# Compute mean and SEM for metrics (e.g., AICR)
mean_metric_train = np.mean(metrics_train, axis=0)
sem_metric_train = np.std(metrics_train, axis=0) / np.sqrt(metrics_train.shape[0])
mean_metric_val = np.mean(metrics_val, axis=0)
sem_metric_val = np.std(metrics_val, axis=0) / np.sqrt(metrics_val.shape[0])

# Plot Loss Curves with Mean ± SEM
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(
        epochs, mean_loss_train, yerr=sem_loss_train, label="Train Mean ± SEM"
    )
    axes[1].errorbar(epochs, mean_loss_val, yerr=sem_loss_val, label="Val Mean ± SEM")
    axes[0].set_title("Training Loss (Mean ± SEM)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss (Mean ± SEM)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Synthetic dataset Loss Curves\nLeft: Training, Right: Validation (Mean ± SEM)"
    )
    fig.savefig(os.path.join(working_dir, "synthetic_loss_mean_sem.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss mean+SEM plot: {e}")
    plt.close()

# Plot AICR Curves with Mean ± SEM
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(
        epochs, mean_metric_train, yerr=sem_metric_train, label="Train Mean ± SEM"
    )
    axes[1].errorbar(
        epochs, mean_metric_val, yerr=sem_metric_val, label="Val Mean ± SEM"
    )
    axes[0].set_title("Training AICR (Mean ± SEM)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AICR")
    axes[1].set_title("Validation AICR (Mean ± SEM)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AICR")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Synthetic dataset AICR Curves\nLeft: Training, Right: Validation (Mean ± SEM)"
    )
    fig.savefig(os.path.join(working_dir, "synthetic_AICR_mean_sem.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating AICR mean+SEM plot: {e}")
    plt.close()
