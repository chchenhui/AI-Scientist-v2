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

# Extract synthetic sweep results
data = experiment_data["embed_dim_sweep"]["synthetic"]
embed_dims = data["embed_dims"]
train_losses = data["losses"]["train"]
val_losses = data["losses"]["val"]
train_ratios = data["metrics"]["train"]
val_ratios = data["metrics"]["val"]
preds = data["predictions"]
gt = data["ground_truth"]

# Plot 1: Loss curves
try:
    plt.figure()
    epochs = np.arange(1, train_losses.shape[1] + 1)
    for i, ed in enumerate(embed_dims):
        plt.plot(epochs, train_losses[i], label=f"train_{ed}")
        plt.plot(epochs, val_losses[i], "--", label=f"val_{ed}")
    plt.title("Synthetic Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Attention retention curves
try:
    plt.figure()
    epochs = np.arange(1, train_ratios.shape[1] + 1)
    for i, ed in enumerate(embed_dims):
        plt.plot(epochs, train_ratios[i], label=f"train_{ed}")
        plt.plot(epochs, val_ratios[i], "--", label=f"val_{ed}")
    plt.title("Synthetic Attention Retention Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_ratio_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Identify smallest and largest embed_dims
idx_min = int(np.argmin(embed_dims))
idx_max = int(np.argmax(embed_dims))

# Plot 3: Predictions for smallest embed_dim
try:
    ed = embed_dims[idx_min]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(gt[idx_min])
    plt.title(f"Ground Truth (embed_dim={ed})")
    plt.subplot(1, 2, 2)
    plt.plot(preds[idx_min])
    plt.title(f"Generated Samples (embed_dim={ed})")
    plt.suptitle(
        f"Left: Ground Truth, Right: Generated Samples (Synthetic, embed_dim={ed})"
    )
    plt.savefig(os.path.join(working_dir, f"synthetic_predictions_ed{ed}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Predictions for largest embed_dim
try:
    ed = embed_dims[idx_max]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(gt[idx_max])
    plt.title(f"Ground Truth (embed_dim={ed})")
    plt.subplot(1, 2, 2)
    plt.plot(preds[idx_max])
    plt.title(f"Generated Samples (embed_dim={ed})")
    plt.suptitle(
        f"Left: Ground Truth, Right: Generated Samples (Synthetic, embed_dim={ed})"
    )
    plt.savefig(os.path.join(working_dir, f"synthetic_predictions_ed{ed}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()
