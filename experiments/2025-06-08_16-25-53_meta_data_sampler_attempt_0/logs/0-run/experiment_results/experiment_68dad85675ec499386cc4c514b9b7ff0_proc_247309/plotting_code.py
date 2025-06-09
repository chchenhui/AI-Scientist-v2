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

# Print summary metrics
for name, data in experiment_data.items():
    try:
        tr_loss = data["losses"]["train"][-1]
        val_loss = data["losses"]["val"][-1]
        tr_acc = data["metrics"]["train"][-1]
        val_acc = data["metrics"]["val"][-1]
        print(
            f"{name}: Final Train Loss={tr_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={tr_acc:.4f}, Val Acc={val_acc:.4f}"
        )
    except Exception:
        print(f"Skipping summary for {name} due to missing data")

# Plot per dataset
for name, data in experiment_data.items():
    losses = data.get("losses", {})
    metrics = data.get("metrics", {})
    corrs = data.get("corrs", [])
    preds = data.get("predictions", [])
    tr_losses = losses.get("train", [])
    val_losses = losses.get("val", [])
    tr_accs = metrics.get("train", [])
    val_accs = metrics.get("val", [])
    # Loss curves
    try:
        plt.figure()
        epochs = np.arange(1, len(tr_losses) + 1)
        plt.plot(epochs, tr_losses, label="Train Loss")
        plt.plot(epochs, val_losses, linestyle="--", label="Val Loss")
        plt.suptitle(f"Dataset: {name} Loss Curves")
        plt.title("Solid: Train Loss, Dashed: Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {name} loss curves: {e}")
        plt.close()
    # Accuracy curves
    try:
        plt.figure()
        epochs = np.arange(1, len(tr_accs) + 1)
        plt.plot(epochs, tr_accs, label="Train Acc")
        plt.plot(epochs, val_accs, linestyle="--", label="Val Acc")
        plt.suptitle(f"Dataset: {name} Accuracy Curves")
        plt.title("Solid: Train Accuracy, Dashed: Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {name} accuracy curves: {e}")
        plt.close()
    # Spearman correlation
    try:
        plt.figure()
        steps = np.arange(1, len(corrs) + 1)
        plt.plot(steps, corrs, marker="o")
        plt.suptitle(f"Dataset: {name} Spearman Correlation")
        plt.title("Correlation of DVN Predictions vs True Contributions")
        plt.xlabel("Meta Update Step")
        plt.ylabel("Spearman Corr")
        plt.savefig(os.path.join(working_dir, f"{name}_spearman_corr.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {name} spearman corr plot: {e}")
        plt.close()
    # Scatter of predictions vs ground truth
    try:
        truths = np.array(data.get("ground_truth", []))
        predictions = np.array(preds)
        if truths.size and predictions.size:
            plt.figure()
            plt.scatter(truths, predictions, alpha=0.6)
            mn = min(truths.min(), predictions.min())
            mx = max(truths.max(), predictions.max())
            plt.plot([mn, mx], [mn, mx], color="red", linestyle="--")
            plt.suptitle(f"Dataset: {name} Predictions vs True Contributions")
            plt.title("DVN Predictions vs Ground Truth")
            plt.xlabel("True Contributions")
            plt.ylabel("DVN Predictions")
            plt.savefig(os.path.join(working_dir, f"{name}_pred_vs_true.png"))
            plt.close()
        else:
            plt.close()
    except Exception as e:
        print(f"Error creating {name} pred vs true scatter: {e}")
        plt.close()

# Comparison plot of validation accuracy
try:
    plt.figure()
    for name, data in experiment_data.items():
        val_accs = data.get("metrics", {}).get("val", [])
        if val_accs:
            epochs = np.arange(1, len(val_accs) + 1)
            plt.plot(epochs, val_accs, marker="o", label=name)
    plt.suptitle("Comparison of Validation Accuracy Across Datasets")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "comparison_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison accuracy plot: {e}")
    plt.close()
