import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot 1: Classifier loss curve
try:
    plt.figure()
    losses = data["synthetic"]["losses"]
    epochs = np.arange(1, len(losses["train"]) + 1)
    plt.plot(epochs, losses["train"], marker="o", label="Train Loss")
    plt.plot(epochs, losses["val"], marker="s", label="Validation Loss")
    plt.title(
        "Synthetic dataset: Classifier Loss Curve\nLeft: Train Loss, Right: Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating classifier loss curve: {e}")
    plt.close()

# Plot 2: DVN Spearman correlation vs epoch
try:
    plt.figure()
    corr = data["synthetic"]["metrics"]["train"]
    epochs = np.arange(1, len(corr) + 1)
    plt.plot(epochs, corr, marker="o")
    plt.title("Synthetic dataset: DVN Spearman Correlation vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.savefig(os.path.join(working_dir, "synthetic_dvn_correlation.png"))
    plt.close()
except Exception as e:
    print(f"Error creating DVN correlation plot: {e}")
    plt.close()

# Plot 3: Predictions vs Ground Truth scatter
try:
    plt.figure()
    preds = np.array(data["synthetic"]["predictions"])
    gt = np.array(data["synthetic"]["ground_truth"])
    plt.scatter(gt, preds, alpha=0.7)
    plt.title("Synthetic dataset: DVN Predictions vs Ground Truth")
    plt.xlabel("Ground Truth Contributions")
    plt.ylabel("Predicted Contributions")
    plt.savefig(os.path.join(working_dir, "synthetic_pred_vs_gt.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
