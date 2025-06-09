import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ds = data["meta_sample_tuning"]["synthetic"]
    metas = ds["meta_sample_values"]
    train_losses = ds["metrics"]["train"]
    val_losses = ds["metrics"]["val"]
    corrs = ds["corr"]
    preds = ds["predictions"]
    truths = ds["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    train_losses = val_losses = corrs = preds = truths = metas = []

# Plot training & validation loss curves
try:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for m, tl in zip(metas, train_losses):
        plt.plot(range(len(tl)), tl, label=f"meta={m}")
    plt.title("Training Loss Curves\nSynthetic Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for m, vl in zip(metas, val_losses):
        plt.plot(range(len(vl)), vl, label=f"meta={m}")
    plt.title("Validation Loss Curves\nSynthetic Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.suptitle("Loss vs Epoch for Meta-sample Tuning")
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot Spearman correlation curves
try:
    plt.figure()
    for m, cr in zip(metas, corrs):
        plt.plot(range(len(cr)), cr, marker="o", label=f"meta={m}")
    plt.title("Spearman Correlation vs Epoch\nSynthetic Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman ρ")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_corr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating spearman correlation plot: {e}")
    plt.close()

# Scatter predicted vs ground truth at final epoch for largest meta-sample
try:
    if metas:
        idx = len(metas) - 1
        pred_last = preds[idx][-1]
        truth_last = truths[idx][-1]
        plt.figure()
        plt.scatter(truth_last, pred_last, alpha=0.7)
        plt.title(f"Pred vs Truth at Final Epoch\nSynthetic Dataset, meta={metas[idx]}")
        plt.xlabel("Ground Truth ΔVal")
        plt.ylabel("Predicted ΔVal")
        fname = f"synthetic_pred_vs_truth_meta{metas[idx]}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()
