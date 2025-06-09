import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper for Spearman correlation
def spearman_corr(a, b):
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return np.corrcoef(ar, br)[0, 1]


# Plot train/val loss per learning rate (up to 5)
lr_data = experiment_data.get("hyperparam_tuning_main_model_lr", {}).get(
    "synthetic", {}
)
for idx, (lr_key, data) in enumerate(lr_data.items()):
    if idx >= 5:
        break
    try:
        plt.figure()
        epochs = np.arange(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Validation Loss")
        plt.title("synthetic dataset Loss Curves\nTrain vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"synthetic_train_val_loss_{lr_key}.png")
        plt.savefig(fname)
    except Exception as e:
        print(f"Error creating loss plot for {lr_key}: {e}")
    finally:
        plt.close()

# Plot Spearman correlation vs epoch for all learning rates
try:
    plt.figure()
    for lr_key, data in lr_data.items():
        corrs = [
            spearman_corr(p, t)
            for p, t in zip(data.get("predictions", []), data.get("ground_truth", []))
        ]
        epochs = np.arange(1, len(corrs) + 1)
        plt.plot(epochs, corrs, label=lr_key)
    plt.title(
        "synthetic dataset Spearman Correlation\nDVN Predictions vs True Contrasts"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Corr")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_spearman_correlation.png")
    plt.savefig(fname)
except Exception as e:
    print(f"Error creating Spearman correlation plot: {e}")
finally:
    plt.close()
