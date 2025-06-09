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

# Plot validation loss comparison
try:
    plt.figure()
    for name, data in experiment_data.items():
        vals = data.get("val_loss", [])
        plt.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=name)
    plt.suptitle("Validation Loss Across Datasets")
    plt.title("Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_loss_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

# Plot validation accuracy comparison
try:
    plt.figure()
    for name, data in experiment_data.items():
        accs = data.get("val_acc", [])
        plt.plot(np.arange(1, len(accs) + 1), accs, marker="s", label=name)
    plt.suptitle("Validation Accuracy Across Datasets")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_accuracy_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation accuracy plot: {e}")
    plt.close()

# Plot Spearman correlation comparison
try:
    plt.figure()
    for name, data in experiment_data.items():
        corrs = data.get("corrs", [])
        plt.plot(np.arange(1, len(corrs) + 1), corrs, marker="^", label=name)
    plt.suptitle("Spearman Correlation of DVN Predictions")
    plt.title("Correlation vs. True Contributions")
    plt.xlabel("Meta‐update Step")
    plt.ylabel("Spearman Correlation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spearman_correlation_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Spearman correlation plot: {e}")
    plt.close()

# Plot N_meta update history comparison
try:
    plt.figure()
    for name, data in experiment_data.items():
        nmeta = data.get("N_meta_history", [])
        plt.plot(np.arange(1, len(nmeta) + 1), nmeta, marker="d", label=name)
    plt.suptitle("Meta‐batch Size (N_meta) History")
    plt.title("Adaptive N_meta over Training")
    plt.xlabel("Meta‐update Step")
    plt.ylabel("N_meta Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "n_meta_history_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating N_meta history plot: {e}")
    plt.close()
