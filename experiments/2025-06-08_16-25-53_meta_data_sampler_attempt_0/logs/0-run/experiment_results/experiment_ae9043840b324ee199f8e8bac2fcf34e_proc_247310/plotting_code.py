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

# 1. Validation Loss Comparison
try:
    plt.figure()
    for name, data in experiment_data.items():
        loss = data.get("val_loss", [])
        epochs = np.arange(1, len(loss) + 1)
        plt.plot(epochs, loss, marker="o", label=name)
    plt.suptitle("Validation Loss Comparison")
    plt.title("Loss vs Epoch for all datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_loss_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss comparison plot: {e}")
    plt.close()

# 2. Validation Accuracy Comparison
try:
    plt.figure()
    for name, data in experiment_data.items():
        acc = data.get("val_acc", [])
        epochs = np.arange(1, len(acc) + 1)
        plt.plot(epochs, acc, marker="s", label=name)
    plt.suptitle("Validation Accuracy Comparison")
    plt.title("Accuracy vs Epoch for all datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_accuracy_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation accuracy comparison plot: {e}")
    plt.close()

# 3. Spearman Correlation Curves
try:
    plt.figure()
    for name, data in experiment_data.items():
        corrs = data.get("corrs", [])
        iters = np.arange(1, len(corrs) + 1)
        plt.plot(iters, corrs, marker="x", label=name)
    plt.suptitle("Spearman Correlation Comparison")
    plt.title("DVN Predictions vs True Contributions")
    plt.xlabel("Iteration")
    plt.ylabel("Spearman Correlation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spearman_correlation_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Spearman correlation comparison plot: {e}")
    plt.close()

# 4. Scatter: Val Loss vs Spearman Corr
try:
    plt.figure()
    for name, data in experiment_data.items():
        loss = data.get("val_loss", [])
        corrs = data.get("corrs", [])
        n = min(len(loss), len(corrs))
        if n > 0:
            plt.scatter(loss[:n], corrs[:n], label=name)
    plt.suptitle("Validation Loss vs Spearman Correlation")
    plt.xlabel("Validation Loss")
    plt.ylabel("Spearman Correlation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "val_loss_vs_spearman_corr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val-loss vs corr scatter plot: {e}")
    plt.close()

# 5. Bar Chart: Final Validation Accuracy
try:
    plt.figure()
    names = list(experiment_data.keys())
    final_acc = [experiment_data[n].get("val_acc", [0])[-1] for n in names]
    x = np.arange(len(names))
    plt.bar(x, final_acc, tick_label=names)
    plt.suptitle("Final Validation Accuracy per Dataset")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(working_dir, "final_val_accuracy_per_dataset.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar chart: {e}")
    plt.close()
