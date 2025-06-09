import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}
data = experiment_data.get("learning_rate_sweep", {}).get("synthetic", {})
lrs = data.get("lrs", [])
metrics = data.get("metrics", {})
losses = data.get("losses", {})
predictions = data.get("predictions", [])
ground_truth = data.get("ground_truth", np.array([]))
# Determine best learning rate
final_vals = [v[-1] for v in metrics.get("val", [])]
if final_vals:
    best_idx = int(np.argmin(final_vals))
    best_lr = lrs[best_idx]
    best_val_err = final_vals[best_idx]
    print(f"Best learning rate: {best_lr}, final validation error: {best_val_err:.4f}")
# Plot error curves
try:
    epochs = len(metrics.get("train", [[]])[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lr, tr, va in zip(lrs, metrics.get("train", []), metrics.get("val", [])):
        axes[0].plot(range(1, epochs + 1), tr, label=f"lr={lr}")
        axes[1].plot(range(1, epochs + 1), va, label=f"lr={lr}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Relative Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative Error")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Error Curves (Synthetic Dataset)\nLeft: Training Error, Right: Validation Error"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves_lr_sweep.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves plot: {e}")
    plt.close()
# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for lr, trl, vl in zip(lrs, losses.get("train", []), losses.get("val", [])):
        axes[0].plot(range(1, epochs + 1), trl, label=f"lr={lr}")
        axes[1].plot(range(1, epochs + 1), vl, label=f"lr={lr}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(
        "Loss Curves (Synthetic Dataset)\nLeft: Training Loss, Right: Validation Loss"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves_lr_sweep.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()
# Plot sample predictions grid
try:
    n_test = ground_truth.shape[0]
    idxs = np.linspace(0, n_test - 1, num=min(5, n_test), dtype=int)
    fig, axes = plt.subplots(len(idxs), 2, figsize=(12, 3 * len(idxs)))
    for (ax0, ax1), idx in zip(axes, idxs):
        ax0.plot(ground_truth[idx])
        ax0.set_title(f"Ground Truth Sample {idx}")
        ax1.plot(predictions[best_idx][idx])
        ax1.set_title(f"Generated Sample {idx} (lr={best_lr})")
    fig.suptitle(
        "Synthetic Dataset Samples\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_sample_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating sample predictions plot: {e}")
    plt.close()
