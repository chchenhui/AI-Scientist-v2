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

# Extract synthetic cluster‐variation results
cluster_counts = (
    experiment_data.get("CLUSTER_COUNT_VARIATION", {})
    .get("synthetic", {})
    .get("cluster_counts", np.array([]))
)
lr_block = (
    experiment_data.get("CLUSTER_COUNT_VARIATION", {})
    .get("synthetic", {})
    .get("learning_rate", {})
)
lrs = lr_block.get("lrs", np.array([]))
metrics_train = lr_block.get("metrics", {}).get("train", np.array([]))
metrics_val = lr_block.get("metrics", {}).get("val", np.array([]))
losses_train = lr_block.get("losses", {}).get("train", np.array([]))
losses_val = lr_block.get("losses", {}).get("val", np.array([]))

# Plot curves for each cluster count
for idx, k in enumerate(cluster_counts):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Synthetic dataset - cluster_count={k} (Left: WG Accuracy, Right: Loss)"
        )
        epochs = np.arange(metrics_train.shape[2])
        for j, lr in enumerate(lrs):
            axes[0].plot(epochs, metrics_train[idx, j], label=f"train lr={lr:.0e}")
            axes[0].plot(
                epochs, metrics_val[idx, j], linestyle="--", label=f"val lr={lr:.0e}"
            )
            axes[1].plot(epochs, losses_train[idx, j], label=f"train lr={lr:.0e}")
            axes[1].plot(
                epochs, losses_val[idx, j], linestyle="--", label=f"val lr={lr:.0e}"
            )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Worst-group Accuracy")
        axes[0].legend()
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        plt.savefig(os.path.join(working_dir, f"synthetic_cluster{k}_wgacc_loss.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for cluster_count={k}: {e}")
        plt.close()

# Print final validation worst-group accuracies
if metrics_val.size:
    final_val = metrics_val[:, :, -1]
    print("Final validation worst-group accuracy per cluster count & lr:")
    for i, k in enumerate(cluster_counts):
        for j, lr in enumerate(lrs):
            print(f"  Cluster {k}, lr={lr:.0e}: {final_val[i, j]:.3f}")
