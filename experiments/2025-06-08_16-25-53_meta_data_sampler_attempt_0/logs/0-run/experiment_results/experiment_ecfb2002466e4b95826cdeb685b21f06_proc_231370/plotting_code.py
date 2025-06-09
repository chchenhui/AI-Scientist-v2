import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
# Load experiment data
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    # Extract metrics
    train_losses = experiment_data["synthetic"]["metrics"]["train"]
    val_losses = experiment_data["synthetic"]["metrics"]["val"]
    print("Train losses per epoch:", train_losses)
    print("Validation losses per epoch:", val_losses)
    # Compute Spearman correlations
    preds_list = experiment_data["synthetic"]["predictions"]
    true_list = experiment_data["synthetic"]["ground_truth"]

    def spearman_corr(a, b):
        a_rank = np.argsort(np.argsort(a))
        b_rank = np.argsort(np.argsort(b))
        return np.corrcoef(a_rank, b_rank)[0, 1]

    corrs = [spearman_corr(p, t) for p, t in zip(preds_list, true_list)]
    print("Spearman correlations per epoch:", corrs)

    # Plot 1: Loss curves
    try:
        plt.figure()
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Synthetic Dataset: Training and Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # Plot 2: Spearman correlation per epoch
    try:
        plt.figure()
        plt.plot(epochs, corrs, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Spearman Correlation")
        plt.title("Synthetic Dataset: DVN Prediction Correlation per Epoch")
        plt.savefig(os.path.join(working_dir, "synthetic_correlation_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating correlation curve: {e}")
        plt.close()

    # Plot 3: Histogram of last-epoch contributions
    try:
        last_idx = len(preds_list) - 1
        true_vals = true_list[last_idx]
        pred_vals = preds_list[last_idx]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].hist(true_vals, bins=10)
        axes[0].set_title("Ground Truth")
        axes[1].hist(pred_vals, bins=10)
        axes[1].set_title("Predicted Samples")
        fig.suptitle(
            f"Synthetic Dataset: Contribution Distribution (Epoch {last_idx})\n"
            "Left: Ground Truth, Right: Predicted Samples"
        )
        plt.savefig(
            os.path.join(
                working_dir, f"synthetic_contribution_dist_epoch_{last_idx}.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()
