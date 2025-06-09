import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    syn = experiment_data["hyperparam_tuning_type_1"]["synthetic"]
    param_values = syn["param_values"]
    loss_train = syn["losses"]["train"]
    loss_val = syn["losses"]["val"]
    corrs = syn["correlations"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training/validation loss curves
try:
    plt.figure()
    for idx, p in enumerate(param_values):
        epochs = np.arange(1, len(loss_train[idx]) + 1)
        plt.plot(epochs, loss_train[idx], label=f"{p} epochs train")
        plt.plot(epochs, loss_val[idx], linestyle="--", label=f"{p} epochs val")
    plt.suptitle("Synthetic Dataset Training/Validation Loss")
    plt.title("Solid: Training Loss, Dashed: Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot Spearman correlation curves
try:
    plt.figure()
    for idx, p in enumerate(param_values):
        epochs = np.arange(1, len(corrs[idx]) + 1)
        plt.plot(epochs, corrs[idx], marker="o", label=f"{p} epochs")
    plt.suptitle("Synthetic Dataset Spearman Correlation")
    plt.title("Correlation of DVN Predictions vs True Contributions")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Corr")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_corr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Spearman correlation plot: {e}")
    plt.close()
