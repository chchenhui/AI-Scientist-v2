import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    exp_path = os.path.join(os.getcwd(), "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot for each ablation and dataset
for abbr, datasets in experiment_data.items():
    for name, data in datasets.items():
        metrics = data.get("metrics", {})
        losses = data.get("losses", {})
        corrs = data.get("corrs", [])
        nmeta = data.get("N_meta_history", [])
        epochs = list(range(1, len(metrics.get("train", [])) + 1))
        steps = list(range(1, len(corrs) + 1))

        # Accuracy curves
        try:
            plt.figure()
            plt.plot(epochs, metrics["train"], label="Train")
            plt.plot(epochs, metrics["val"], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"Training vs Validation Accuracy\nAblation: {abbr}, Dataset: {name}"
            )
            plt.legend()
            fname = f"{name}_{abbr}_accuracy_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {abbr}-{name}: {e}")
            plt.close()

        # Loss curves
        try:
            plt.figure()
            plt.plot(epochs, losses["train"], label="Train")
            plt.plot(epochs, losses["val"], label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Training vs Validation Loss\nAblation: {abbr}, Dataset: {name}")
            plt.legend()
            fname = f"{name}_{abbr}_loss_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {abbr}-{name}: {e}")
            plt.close()

        # Correlation history
        try:
            plt.figure()
            plt.plot(steps, corrs, marker="o")
            plt.xlabel("Meta Update Step")
            plt.ylabel("Spearman Correlation")
            plt.title(f"Correlation History\nAblation: {abbr}, Dataset: {name}")
            fname = f"{name}_{abbr}_corr_history.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating corr history plot for {abbr}-{name}: {e}")
            plt.close()

        # N_meta history
        try:
            plt.figure()
            plt.plot(steps, nmeta, marker="o")
            plt.xlabel("Meta Update Step")
            plt.ylabel("N_meta")
            plt.title(f"N_meta History\nAblation: {abbr}, Dataset: {name}")
            fname = f"{name}_{abbr}_nmeta_history.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating N_meta history plot for {abbr}-{name}: {e}")
            plt.close()
