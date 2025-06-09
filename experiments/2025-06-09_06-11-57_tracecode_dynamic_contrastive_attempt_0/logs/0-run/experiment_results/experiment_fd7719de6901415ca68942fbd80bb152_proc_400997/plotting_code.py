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

# Print final validation accuracies
for scheme, synth_data in (
    experiment_data.get("tokenization_granularity", {}).get("synthetic", {}).items()
):
    for E, data in synth_data.items():
        try:
            fv = data["metrics"]["val_acc"][-1]
            print(f"Scheme={scheme}, Epochs={E}, Final Val Acc={fv:.4f}")
        except Exception:
            pass

# Plot metrics curves per scheme
for scheme, synth_data in (
    experiment_data.get("tokenization_granularity", {}).get("synthetic", {}).items()
):
    try:
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        fig.suptitle(f"Synthetic Dataset Metrics – Scheme: {scheme}", fontsize=14)
        for E, data in sorted(synth_data.items()):
            epochs = np.arange(1, len(data["losses"]["train"]) + 1)
            axes[0].plot(epochs, data["losses"]["train"], label=f"Train loss E{E}")
            axes[0].plot(epochs, data["losses"]["val"], label=f"Val loss   E{E}")
            axes[1].plot(epochs, data["metrics"]["train_acc"], label=f"Train acc E{E}")
            axes[1].plot(epochs, data["metrics"]["val_acc"], label=f"Val acc   E{E}")
            axes[2].plot(
                epochs, data["metrics"]["train_alignment_gap"], label=f"Train gap E{E}"
            )
            axes[2].plot(
                epochs, data["metrics"]["val_alignment_gap"], label=f"Val gap   E{E}"
            )
        axes[0].set_title("Training vs. Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[1].set_title("Training vs. Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[2].set_title("Training vs. Validation Alignment Gap")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Alignment Gap")
        axes[2].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"synthetic_{scheme}_metrics_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot_{scheme}: {e}")
        plt.close()
