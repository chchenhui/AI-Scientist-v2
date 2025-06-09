import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    exp_file = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract synthetic optimizer results
data = experiment_data["optimizer"]["synthetic"]
names = data["optim_names"]
metrics = data["metrics"]
train_loss = metrics["train_loss"]
val_loss = metrics["val_loss"]
train_AICR = metrics["train_AICR"]
val_AICR = metrics["val_AICR"]

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(train_loss[0]) + 1)
    for i, name in enumerate(names):
        axes[0].plot(epochs, train_loss[i], label=name)
        axes[1].plot(epochs, val_loss[i], label=name)
    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset Loss Curves\nLeft: Training Loss, Right: Validation Loss"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot AICR curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(train_AICR[0]) + 1)
    for i, name in enumerate(names):
        axes[0].plot(epochs, train_AICR[i], label=name)
        axes[1].plot(epochs, val_AICR[i], label=name)
    axes[0].set_title("Training AICR")
    axes[1].set_title("Validation AICR")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("AICR")
    axes[1].set_ylabel("AICR")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Synthetic Dataset AICR Curves\nLeft: Training AICR, Right: Validation AICR"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_AICR_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AICR plot: {e}")
    plt.close()
