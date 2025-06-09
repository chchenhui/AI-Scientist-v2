import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")

# Load and process experiment data
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
    d = experiment_data["meta_update_lr_tuning"]["synthetic"]
    meta_lrs = d["meta_lrs"]
    train_losses = d["losses"]["train"]
    val_losses = d["losses"]["val"]
    preds = d["predictions"]
    truths = d["ground_truth"]

    def spearman_corr(a, b):
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        return np.corrcoef(ar, br)[0, 1]

    spearman = [
        [
            spearman_corr(np.array(preds[i][j]), np.array(truths[i][j]))
            for j in range(len(preds[i]))
        ]
        for i in range(len(meta_lrs))
    ]
except Exception as e:
    print(f"Error loading or processing experiment data: {e}")

# Plot 1: Loss curves for all meta_lrs
try:
    plt.figure()
    for i, lr in enumerate(meta_lrs):
        epochs = np.arange(1, len(train_losses[i]) + 1)
        plt.plot(epochs, train_losses[i], label=f"Train lr={lr:.1e}")
        plt.plot(epochs, val_losses[i], linestyle="--", label=f"Val lr={lr:.1e}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Dataset - Train & Val Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: Final validation loss vs meta_lr
try:
    final_val = [vl[-1] for vl in val_losses]
    plt.figure()
    plt.semilogx(meta_lrs, final_val, marker="o")
    plt.xlabel("MetaLR")
    plt.ylabel("Final Validation Loss")
    plt.title("Synthetic Dataset - Final Val Loss vs MetaLR")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_loss_vs_metalr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final val loss vs meta_lr plot: {e}")
    plt.close()

# Plot 3: Spearman correlation vs epoch for each meta_lr
try:
    plt.figure()
    for i, lr in enumerate(meta_lrs):
        epochs = np.arange(1, len(spearman[i]) + 1)
        plt.plot(epochs, spearman[i], marker="o", label=f"lr={lr:.1e}")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.title("Synthetic Dataset - Spearman Corr vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_vs_epoch.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Spearman correlation plot: {e}")
    plt.close()
