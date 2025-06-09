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

data = experiment_data.get("synthetic", {})
losses = data.get("losses", {})
metrics = data.get("metrics", {})
train_loss = losses.get("train", [])
val_loss = losses.get("val", [])
train_metric = metrics.get("train", [])
val_metric = metrics.get("val", [])
preds = data.get("predictions", np.array([]))
truths = data.get("ground_truth", np.array([]))
epochs = list(range(1, len(train_loss) + 1))

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o")
    axes[1].plot(epochs, val_loss, marker="o")
    axes[0].set_title("Left: Training Loss")
    axes[1].set_title("Right: Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    fig.suptitle("Synthetic Dataset Loss Curves")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot weighted‐group accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_metric, marker="o")
    axes[1].plot(epochs, val_metric, marker="o")
    axes[0].set_title("Left: Training Weighted Group Accuracy")
    axes[1].set_title("Right: Validation Weighted Group Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle("Synthetic Dataset Weighted Group Accuracy")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating wg accuracy plot: {e}")
    plt.close()

# Plot class distribution of ground truth vs predictions
try:
    ut, ct = np.unique(truths, return_counts=True)
    up, cp = np.unique(preds, return_counts=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(ut, ct)
    axes[1].bar(up, cp)
    axes[0].set_title("Left: Ground Truth Distribution")
    axes[1].set_title("Right: Prediction Distribution")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")
    fig.suptitle("Synthetic Dataset Class Distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_class_distribution.png"))
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()
