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

datasets = list(experiment_data.get("fixed_N_meta", {}).keys())

# Plot 1: Accuracy Curves
try:
    plt.figure()
    for name in datasets:
        acc_tr = experiment_data["fixed_N_meta"][name]["metrics"]["train"]
        acc_val = experiment_data["fixed_N_meta"][name]["metrics"]["val"]
        epochs = range(1, len(acc_tr) + 1)
        plt.plot(epochs, acc_tr, marker="o", label=f"{name} Train")
        plt.plot(epochs, acc_val, marker="x", linestyle="--", label=f"{name} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.suptitle("Training and Validation Accuracy")
    plt.title("All Datasets")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "accuracy_all_datasets.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 2: Loss Curves
try:
    plt.figure()
    for name in datasets:
        loss_tr = experiment_data["fixed_N_meta"][name]["losses"]["train"]
        loss_val = experiment_data["fixed_N_meta"][name]["losses"]["val"]
        epochs = range(1, len(loss_tr) + 1)
        plt.plot(epochs, loss_tr, marker="o", label=f"{name} Train")
        plt.plot(epochs, loss_val, marker="x", linestyle="--", label=f"{name} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.suptitle("Training and Validation Loss")
    plt.title("All Datasets")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_all_datasets.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 3: Spearman Correlation over Meta-update Steps
try:
    plt.figure()
    for name in datasets:
        corrs = experiment_data["fixed_N_meta"][name]["corrs"]
        steps = range(1, len(corrs) + 1)
        plt.plot(steps, corrs, marker="o", label=name)
    plt.xlabel("Meta-update Step")
    plt.ylabel("Spearman Correlation")
    plt.suptitle("Spearman Correlation over Meta-update Steps")
    plt.title("All Datasets")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spearman_corr_all.png"))
    plt.close()
except Exception as e:
    print(f"Error creating spearman correlation plot: {e}")
    plt.close()

# Plot 4: Predicted vs Ground Truth Scatter
try:
    fig, axs = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
    for ax, name in zip(np.atleast_1d(axs), datasets):
        y_true = experiment_data["fixed_N_meta"][name]["ground_truth"]
        y_pred = experiment_data["fixed_N_meta"][name]["predictions"]
        ax.scatter(y_true, y_pred, alpha=0.6)
        mn, mx = min(y_true), max(y_true)
        ax.plot([mn, mx], [mn, mx], color="red", linestyle="--")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predictions")
        ax.set_title(name)
    fig.suptitle("Predicted vs Ground Truth across Datasets")
    plt.savefig(os.path.join(working_dir, "pred_vs_true_all.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predicted vs ground truth plot: {e}")
    plt.close()
