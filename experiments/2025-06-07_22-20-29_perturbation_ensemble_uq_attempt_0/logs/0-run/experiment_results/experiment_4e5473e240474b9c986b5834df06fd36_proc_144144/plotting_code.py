import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

data = exp.get("adam_beta1", {}).get("synthetic", {})
betas = data.get("beta1_list", [])

# Plot AUC curves
try:
    plt.figure()
    for i, beta in enumerate(betas):
        train_auc = data["metrics"]["train"][i]
        val_auc = data["metrics"]["val"][i]
        epochs = range(1, len(train_auc) + 1)
        plt.plot(epochs, train_auc, label=f"Train AUC β1={beta}")
        plt.plot(epochs, val_auc, "--", label=f"Val AUC β1={beta}")
    plt.title("AUC Curves over Epochs\nDataset: synthetic")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_auc_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AUC curves plot: {e}")
    plt.close()

# Plot loss curves
try:
    plt.figure()
    for i, beta in enumerate(betas):
        train_loss = data["losses"]["train"][i]
        val_loss = data["losses"]["val"][i]
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label=f"Train Loss β1={beta}")
        plt.plot(epochs, val_loss, "--", label=f"Val Loss β1={beta}")
    plt.title("Loss Curves over Epochs\nDataset: synthetic")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot ROC curves
try:
    from sklearn.metrics import roc_curve, auc

    plt.figure()
    for i, beta in enumerate(betas):
        preds = data["predictions"][i]
        labels = data["ground_truth"][i]
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"β1={beta} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.title("ROC Curves for Final Predictions\nDataset: synthetic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_roc_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ROC curves plot: {e}")
    plt.close()
