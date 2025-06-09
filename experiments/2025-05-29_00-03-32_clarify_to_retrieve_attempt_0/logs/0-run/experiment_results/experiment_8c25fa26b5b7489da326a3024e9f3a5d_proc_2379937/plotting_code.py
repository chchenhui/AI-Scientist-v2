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

lr_sweep = experiment_data.get("learning_rate_sweep", {}).get("synthetic_xor", {})
lr_keys = sorted(lr_sweep.keys(), key=lambda k: float(k.split("_")[1]))

# Plot loss curves
try:
    plt.figure()
    for k in lr_keys:
        tr = lr_sweep[k]["losses"]["train"]
        va = lr_sweep[k]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{k} train")
        plt.plot(epochs, va, "--", label=f"{k} val")
    plt.title("Loss Curves (synthetic_xor)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot CES metric curves
try:
    plt.figure()
    for k in lr_keys:
        trc = lr_sweep[k]["metrics"]["train"]
        vac = lr_sweep[k]["metrics"]["val"]
        epochs = range(1, len(trc) + 1)
        plt.plot(epochs, trc, label=f"{k} train CES")
        plt.plot(epochs, vac, "--", label=f"{k} val CES")
    plt.title("CES Metric Curves (synthetic_xor)")
    plt.xlabel("Epoch")
    plt.ylabel("CES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_xor_CES_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES metric plot: {e}")
    plt.close()

# Identify best LR by final val CES and print metrics
best_lr = None
best_val_ces = -np.inf
for k in lr_keys:
    final_ces = lr_sweep[k]["metrics"]["val"][-1]
    print(f"LR {k}: final_val_CES={final_ces:.4f}")
    if final_ces > best_val_ces:
        best_val_ces, best_lr = final_ces, k
if best_lr:
    print(f"Best LR: {best_lr}, Final Val CES: {best_val_ces:.4f}")

# Plot ground truth vs predictions for best LR
try:
    if best_lr:
        gt = lr_sweep[best_lr]["ground_truth"][-1]
        preds = lr_sweep[best_lr]["predictions"][-1]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(gt, bins=np.arange(3) - 0.5)
        plt.title("Ground Truth")
        plt.subplot(1, 2, 2)
        plt.hist(preds, bins=np.arange(3) - 0.5)
        plt.title("Predictions")
        plt.suptitle(
            "Ground Truth vs Predictions (synthetic_xor) - Left: Ground Truth, Right: Predictions"
        )
        plt.savefig(
            os.path.join(working_dir, "synthetic_xor_ground_truth_vs_predictions.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating ground truth vs predictions plot: {e}")
    plt.close()
