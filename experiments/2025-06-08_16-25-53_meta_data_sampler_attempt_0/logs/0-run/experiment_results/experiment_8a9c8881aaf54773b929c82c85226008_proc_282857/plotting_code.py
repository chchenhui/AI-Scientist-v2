import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot accuracy and loss per dataset
for dataset in next(iter(experiment_data.values())).keys():
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for ablation, style in zip(experiment_data.keys(), ["-", "--"]):
            metrics = experiment_data[ablation][dataset]["metrics"]
            losses = experiment_data[ablation][dataset]["losses"]
            epochs = range(1, len(metrics["train"]) + 1)
            ax1.plot(epochs, metrics["train"], style, label=f"{ablation} train")
            ax1.plot(epochs, metrics["val"], style, label=f"{ablation} val")
            ax2.plot(epochs, losses["train"], style, label=f"{ablation} train")
            ax2.plot(epochs, losses["val"], style, label=f"{ablation} val")
        ax1.set_title("Training/Validation Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax2.set_title("Training/Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        fig.suptitle(f"{dataset} Performance Curves")
        plt.savefig(os.path.join(working_dir, f"{dataset}_perf_curves.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating performance curves for {dataset}: {e}")
        plt.close()

# Plot correlation history per ablation
for ablation in experiment_data.keys():
    try:
        fig = plt.figure(figsize=(6, 4))
        for dataset in experiment_data[ablation].keys():
            corrs = experiment_data[ablation][dataset].get("corrs", [])
            updates = range(1, len(corrs) + 1)
            if corrs:
                plt.plot(updates, corrs, label=dataset)
        plt.title(f"Spearman Correlation over Meta-Updates ({ablation})")
        plt.xlabel("Meta-update Index")
        plt.ylabel("Spearman ρ")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"corr_history_{ablation}.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating correlation plot for {ablation}: {e}")
        plt.close()
