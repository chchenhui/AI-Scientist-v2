import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["triplet_margin_ablation"]["synthetic"]
    margins = sorted(data.keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    margins = []

# Plot loss & accuracy curves for each margin
for m in margins:
    try:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Synthetic Dataset: Triplet Margin {m}")
        epochs = range(1, len(data[m]["losses"]["train"]) + 1)
        # Loss curves
        axs[0].plot(epochs, data[m]["losses"]["train"], label="Train")
        axs[0].plot(epochs, data[m]["losses"]["val"], label="Val")
        axs[0].set_title("Left: Loss curves")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        # Accuracy curves
        axs[1].plot(epochs, data[m]["metrics"]["train"], label="Train")
        axs[1].plot(epochs, data[m]["metrics"]["val"], label="Val")
        axs[1].set_title("Right: Accuracy curves")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        fig.savefig(os.path.join(working_dir, f"synthetic_margin_{m}_curves.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for margin={m}: {e}")
        plt.close()

# Summary plot of final validation accuracy vs. margin
try:
    final_vals = [data[m]["metrics"]["val"][-1] for m in margins]
    plt.figure()
    plt.plot(margins, final_vals, marker="o")
    plt.title("Synthetic Dataset: Final Validation Accuracy vs Triplet Margin")
    plt.xlabel("Margin")
    plt.ylabel("Validation Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()
