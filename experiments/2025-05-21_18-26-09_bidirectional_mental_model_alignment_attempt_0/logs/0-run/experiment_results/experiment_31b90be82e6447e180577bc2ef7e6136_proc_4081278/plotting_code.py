import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

if "experiment_data" in locals():
    datasets = list(next(iter(experiment_data.values())).keys())
    activations = list(experiment_data.keys())
else:
    datasets, activations = [], []

for dataset in datasets:
    try:
        plt.figure(figsize=(10, 5))
        # Left: Loss Curves
        ax1 = plt.subplot(1, 2, 1)
        for act in activations:
            losses = experiment_data[act][dataset]["losses"]
            epochs = range(1, len(losses["train"]) + 1)
            ax1.plot(epochs, losses["train"], label=f"{act} Train")
            ax1.plot(epochs, losses["val"], linestyle="--", label=f"{act} Val")
        ax1.set_title("Training vs Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Right: MAI Curves
        ax2 = plt.subplot(1, 2, 2)
        for act in activations:
            mai = experiment_data[act][dataset]["mai"]
            ax2.plot(range(1, len(mai) + 1), mai, label=act)
        ax2.set_title("MAI over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAI")
        ax2.legend()

        plt.suptitle(
            f"{dataset} Metrics (Left: Loss Curves, Right: MAI Curves)", fontsize=14
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(working_dir, f"{dataset}_metrics.png")
        plt.savefig(save_path)
        plt.close()

        # Print final MAI values
        final_mai = {
            act: experiment_data[act][dataset]["mai"][-1] for act in activations
        }
        print(f"{dataset} Final MAI: {final_mai}")
    except Exception as e:
        print(f"Error creating plot for {dataset}: {e}")
        plt.close()
