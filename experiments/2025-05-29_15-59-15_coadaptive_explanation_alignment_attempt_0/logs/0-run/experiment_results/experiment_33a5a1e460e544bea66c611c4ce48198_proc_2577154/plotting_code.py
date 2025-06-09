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

for scenario, combos in experiment_data.get("teacher_feature_removal", {}).items():
    for combo_key, res in combos.items():
        try:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            epochs = np.arange(1, len(res["metrics"]["train"]) + 1)
            # Left: Accuracy
            axs[0].plot(epochs, res["metrics"]["train"], label="Train Acc")
            axs[0].plot(epochs, res["metrics"]["val"], label="Val Acc")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Accuracy")
            axs[0].set_title("Left: Train vs Val Accuracy")
            axs[0].legend()
            # Right: Loss
            axs[1].plot(epochs, res["losses"]["train"], label="Train Loss")
            axs[1].plot(epochs, res["losses"]["val"], label="Val Loss")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Loss")
            axs[1].set_title("Right: Train vs Val Loss")
            axs[1].legend()
            fig.suptitle(f"{scenario} | {combo_key}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fname = f"{scenario}_{combo_key}_metrics.png".replace(" ", "_")
            fig.savefig(os.path.join(working_dir, fname))
            plt.close(fig)
        except Exception as e:
            print(f"Error creating metrics plot for {scenario} {combo_key}: {e}")
            plt.close("all")
