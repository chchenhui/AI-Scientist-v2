import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Experiment data paths under AI_SCIENTIST_ROOT
experiment_data_path_list = [
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/"
    "experiment_results/experiment_f2502cda9c434cd0a2df81e1d471f9a3_proc_4194300/experiment_data.npy",
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/"
    "experiment_results/experiment_763d4f6c26f84e40a5dfc2995da55e62_proc_4194299/experiment_data.npy",
    "experiments/2025-06-06_23-36-12_gradient_cluster_robust_attempt_0/logs/0-run/"
    "experiment_results/experiment_cfffc902aaa84d0caed31d045775cadb_proc_4194301/experiment_data.npy",
]

# Load and collect synthetic data from all experiments
all_data = []
try:
    for rel_path in experiment_data_path_list:
        exp = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), rel_path), allow_pickle=True
        ).item()
        all_data.append(exp.get("synthetic", {}))
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract lists of curves and predictions
train_losses_list = [d.get("losses", {}).get("train", []) for d in all_data]
val_losses_list = [d.get("losses", {}).get("val", []) for d in all_data]
train_metrics_list = [d.get("metrics", {}).get("train", []) for d in all_data]
val_metrics_list = [d.get("metrics", {}).get("val", []) for d in all_data]
preds_list = [d.get("predictions", np.array([])) for d in all_data]
truths_list = [d.get("ground_truth", np.array([])) for d in all_data]

# Determine common epoch range
min_epochs = min([len(x) for x in train_losses_list + val_losses_list] or [0])
epochs = np.arange(1, min_epochs + 1)

# Convert and crop to arrays
train_losses = np.array([lst[:min_epochs] for lst in train_losses_list])
val_losses = np.array([lst[:min_epochs] for lst in val_losses_list])
train_metrics = np.array([lst[:min_epochs] for lst in train_metrics_list])
val_metrics = np.array([lst[:min_epochs] for lst in val_metrics_list])


# Function to compute mean and standard error
def mean_sem(arr):
    m = np.mean(arr, axis=0)
    s = (
        np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(m)
    )
    return m, s


train_loss_mean, train_loss_sem = mean_sem(train_losses)
val_loss_mean, val_loss_sem = mean_sem(val_losses)
train_acc_mean, train_acc_sem = mean_sem(train_metrics)
val_acc_mean, val_acc_sem = mean_sem(val_metrics)

# Plot loss curves with error bars
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(
        epochs,
        train_loss_mean,
        yerr=train_loss_sem,
        marker="o",
        label="Train Loss Mean ± SE",
    )
    axes[1].errorbar(
        epochs, val_loss_mean, yerr=val_loss_sem, marker="o", label="Val Loss Mean ± SE"
    )
    axes[0].set_title("Left: Training Loss Mean ± SE")
    axes[1].set_title("Right: Validation Loss Mean ± SE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("Synthetic Dataset Loss Curves (Mean ± SE)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot weighted-group accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(
        epochs,
        train_acc_mean,
        yerr=train_acc_sem,
        marker="o",
        label="Train Accuracy Mean ± SE",
    )
    axes[1].errorbar(
        epochs,
        val_acc_mean,
        yerr=val_acc_sem,
        marker="o",
        label="Val Accuracy Mean ± SE",
    )
    axes[0].set_title("Left: Training Weighted Group Accuracy Mean ± SE")
    axes[1].set_title("Right: Validation Weighted Group Accuracy Mean ± SE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    fig.suptitle("Synthetic Dataset Weighted Group Accuracy (Mean ± SE)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy_curves_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating wg accuracy plot: {e}")
    plt.close()

# Plot class distribution mean and SE
try:
    # Combine labels across experiments
    all_truths = np.concatenate(truths_list) if truths_list else np.array([])
    all_preds = np.concatenate(preds_list) if preds_list else np.array([])
    labels = (
        np.unique(np.concatenate([all_truths, all_preds]))
        if all_truths.size or all_preds.size
        else np.array([])
    )
    truth_counts = np.array(
        [[np.sum(trut == lbl) for lbl in labels] for trut in truths_list]
    )
    pred_counts = np.array(
        [[np.sum(pred == lbl) for lbl in labels] for pred in preds_list]
    )
    truth_mean, truth_sem = mean_sem(truth_counts)
    pred_mean, pred_sem = mean_sem(pred_counts)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, truth_mean, yerr=truth_sem, capsize=5)
    axes[1].bar(labels, pred_mean, yerr=pred_sem, capsize=5)
    axes[0].set_title("Left: Ground Truth Distribution Mean ± SE")
    axes[1].set_title("Right: Prediction Distribution Mean ± SE")
    for ax in axes:
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
    fig.suptitle("Synthetic Dataset Class Distribution (Mean ± SE)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_class_distribution_mean_se.png"))
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()
