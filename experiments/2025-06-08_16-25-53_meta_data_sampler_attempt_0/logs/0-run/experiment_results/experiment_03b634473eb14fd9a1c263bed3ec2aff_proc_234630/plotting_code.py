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

# Extract synthetic dataset results
sd = experiment_data.get("batch_size", {}).get("synthetic", {})
batch_sizes = sd.get("batch_sizes", [])
train_metrics = sd.get("metrics", {}).get("train", [])
val_metrics = sd.get("metrics", {}).get("val", [])
predictions = sd.get("predictions", [])
ground_truth = sd.get("ground_truth", [])

# Plot training and validation loss curves
try:
    plt.figure()
    epochs = range(1, len(train_metrics[0]) + 1) if train_metrics else []
    for i, bsz in enumerate(batch_sizes):
        plt.plot(epochs, train_metrics[i], label=f"Train bs={bsz}")
        plt.plot(epochs, val_metrics[i], "--", label=f"Val bs={bsz}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Dataset Loss Curves\nTraining and Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()


# Define Spearman correlation
def spearman_corr(a, b):
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return np.corrcoef(ar, br)[0, 1]


# Plot DVN Spearman correlations
try:
    plt.figure()
    epochs = range(1, len(predictions[0]) + 1) if predictions else []
    for i, bsz in enumerate(batch_sizes):
        corrs = []
        for ep in range(len(predictions[i])):
            preds = np.array(predictions[i][ep])
            trues = np.array(ground_truth[i][ep])
            corrs.append(spearman_corr(preds, trues))
        plt.plot(epochs, corrs, marker="o", label=f"bs={bsz}")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.title(
        "Synthetic Dataset DVN Spearman Correlation\nMeta-sample Predictions vs True Improvement"
    )
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_spearman_corr.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()
