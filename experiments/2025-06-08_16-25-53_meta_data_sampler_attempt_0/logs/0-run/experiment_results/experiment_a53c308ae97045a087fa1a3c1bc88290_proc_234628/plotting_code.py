import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# 1) Loss curves
try:
    plt.figure()
    syn = experiment_data["DVN_LR"]["synthetic"]
    for lr, exp in syn.items():
        epochs = np.arange(len(exp["metrics"]["train"]))
        plt.plot(epochs, exp["metrics"]["train"], label=f"{lr}-train")
        plt.plot(epochs, exp["metrics"]["val"], label=f"{lr}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic dataset - Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()


# 2) Spearman correlation per epoch
def spearman_corr(a, b):
    ar, br = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ar, br)[0, 1]


try:
    plt.figure()
    for lr, exp in syn.items():
        corrs = [
            spearman_corr(p, g) for p, g in zip(exp["predictions"], exp["ground_truth"])
        ]
        plt.plot(range(len(corrs)), corrs, label=f"LR {lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.title("Synthetic dataset - Spearman Correlation per Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_per_epoch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating spearman correlation plot: {e}")
    plt.close()

# 3) Scatter of predictions vs ground truth at last epoch
try:
    lrs = list(syn.keys())
    fig, axes = plt.subplots(1, len(lrs), figsize=(5 * len(lrs), 5))
    for ax, lr in zip(np.atleast_1d(axes).flatten(), lrs):
        preds = syn[lr]["predictions"][-1]
        gt = syn[lr]["ground_truth"][-1]
        ax.scatter(gt, preds)
        ax.set_xlabel("Ground Truth Contrast")
        ax.set_ylabel("Predicted Contrast")
        ax.set_title(f"LR {lr}")
    fig.suptitle(
        f"Synthetic dataset - Predictions vs Ground Truth (Last Epoch)\n"
        f"Subplots: Left: LR {lrs[0]}, Center: LR {lrs[1]}, Right: LR {lrs[-1]}"
        if len(lrs) == 3
        else ""
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_pred_vs_gt_last_epoch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
