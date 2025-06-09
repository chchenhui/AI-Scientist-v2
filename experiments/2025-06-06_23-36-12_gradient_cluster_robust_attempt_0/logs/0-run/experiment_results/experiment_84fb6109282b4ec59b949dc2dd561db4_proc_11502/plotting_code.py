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

datasets = ["mnist", "fashion_mnist", "cifar10"]
val_wg = {d: np.array(experiment_data[d]["metrics"]["val_wg"]) for d in datasets}
nmi = {d: np.array(experiment_data[d]["nmi"]) for d in datasets}
epochs = np.arange(1, next(iter(val_wg.values())).shape[0] + 1)

# Comparison plot: Validation worst-group accuracy
try:
    plt.figure()
    for d in datasets:
        plt.plot(epochs, val_wg[d], marker="o", label=d)
    plt.title("Validation Worst-group Accuracy\nComparison across datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Worst-group Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "comparison_validation_wg_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison_validation_wg_accuracy: {e}")
    plt.close()

# Comparison plot: NMI curves
try:
    plt.figure()
    for d in datasets:
        plt.plot(epochs, nmi[d], marker="o", label=d)
    plt.title("NMI over Epochs\nComparison across datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Mutual Information")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "comparison_nmi_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison_nmi_curves: {e}")
    plt.close()

# Per-dataset plots: Validation WG and NMI side by side
for d in datasets:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, val_wg[d], marker="o")
        axes[0].set_title(f"Validation WG Accuracy\nDataset: {d}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Worst-group Accuracy")
        axes[1].plot(epochs, nmi[d], marker="o")
        axes[1].set_title(f"NMI over Epochs\nDataset: {d}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Normalized Mutual Information")
        fig.suptitle(f"{d} - Validation WG and NMI")
        plt.savefig(os.path.join(working_dir, f"{d}_validation_wg_nmi.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating {d}_validation_wg_nmi: {e}")
        plt.close("all")
