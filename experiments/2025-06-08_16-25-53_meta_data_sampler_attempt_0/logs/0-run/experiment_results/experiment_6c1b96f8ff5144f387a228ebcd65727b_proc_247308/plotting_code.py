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

# Extract per-dataset metrics
datasets = list(experiment_data.keys())
losses_train = {d: experiment_data[d]["losses"]["train"] for d in datasets}
losses_val = {d: experiment_data[d]["losses"]["val"] for d in datasets}
metrics_train = {d: experiment_data[d]["metrics"]["train"] for d in datasets}
metrics_val = {d: experiment_data[d]["metrics"]["val"] for d in datasets}
corrs = {d: experiment_data[d]["corrs"] for d in datasets}

# Print final accuracies
for d in datasets:
    if metrics_train[d] and metrics_val[d]:
        print(
            f"[{d}] Final Train Acc = {metrics_train[d][-1]:.4f}, Final Val Acc = {metrics_val[d][-1]:.4f}"
        )

# Combined loss curves
try:
    plt.figure()
    for d in datasets:
        epochs = np.arange(1, len(losses_train[d]) + 1)
        plt.plot(epochs, losses_train[d], label=f"{d} train")
        plt.plot(epochs, losses_val[d], linestyle="--", label=f"{d} val")
    plt.suptitle("Training/Validation Loss Curves Across Datasets")
    plt.title("Solid: Training, Dashed: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Combined accuracy curves
try:
    plt.figure()
    for d in datasets:
        epochs = np.arange(1, len(metrics_train[d]) + 1)
        plt.plot(epochs, metrics_train[d], label=f"{d} train")
        plt.plot(epochs, metrics_val[d], linestyle="--", label=f"{d} val")
    plt.suptitle("Training/Validation Accuracy Across Datasets")
    plt.title("Solid: Training, Dashed: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# Combined Spearman correlation curves
try:
    plt.figure()
    for d in datasets:
        steps = np.arange(1, len(corrs[d]) + 1)
        plt.plot(steps, corrs[d], marker="o", label=d)
    plt.suptitle("DVN Spearman Correlations Across Datasets")
    plt.title("Correlation of Predicted vs True Contributions")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Spearman Corr")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "all_datasets_spearman_corr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Spearman correlation plot: {e}")
    plt.close()

# Final accuracy comparison bar chart
try:
    plt.figure()
    x = np.arange(len(datasets))
    final_train = [metrics_train[d][-1] for d in datasets]
    final_val = [metrics_val[d][-1] for d in datasets]
    width = 0.35
    plt.bar(x - width / 2, final_train, width, label="Train")
    plt.bar(x + width / 2, final_val, width, label="Val")
    plt.xticks(x, datasets)
    plt.suptitle("Final Train/Validation Accuracies by Dataset")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "final_accuracies_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy comparison: {e}")
    plt.close()
