import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data and summarize
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    sd = exp["softmax_temperature"]["synthetic"]
    T_vals = sd["T_values"]
    train_losses = sd["losses"]["train"]
    val_losses = sd["losses"]["val"]
    correlations = sd["correlation"]
    preds = sd["predictions"]
    gts = sd["ground_truth"]
    print("Final metrics by T:")
    for T, tr, va, corr in zip(T_vals, train_losses, val_losses, correlations):
        print(
            f" T={T}: Train Loss={tr[-1]:.4f}, Val Loss={va[-1]:.4f}, DVN Spearman={corr[-1]:.4f}"
        )
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot train/val loss curves
try:
    plt.figure()
    for T, tr, va in zip(T_vals, train_losses, val_losses):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"Train T={T}")
        plt.plot(epochs, va, "--", label=f"Val T={T}")
    plt.title("Synthetic dataset Loss Curves (Train & Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_loss_curves_all_T.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot DVN Spearman correlation curves
try:
    plt.figure()
    for T, corr in zip(T_vals, correlations):
        epochs = range(1, len(corr) + 1)
        plt.plot(epochs, corr, marker="o", label=f"T={T}")
    plt.title("Synthetic dataset DVN Spearman Correlation vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Corr")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_dvn_correlation_all_T.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating correlation plot: {e}")
    plt.close()

# Scatter of DVN predictions vs ground truth at final epoch
try:
    plt.figure()
    for T, p_list, g_list in zip(T_vals, preds, gts):
        p_last = p_list[-1]
        g_last = g_list[-1]
        plt.scatter(g_last, p_last, label=f"T={T}", alpha=0.7)
    plt.title(
        "Synthetic dataset DVN Predictions vs Ground Truth (Epoch {})".format(
            len(train_losses[0])
        )
    )
    plt.xlabel("True Contribution")
    plt.ylabel("Predicted Contribution")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_pred_vs_gt_last_epoch.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating pred vs gt scatter: {e}")
    plt.close()
