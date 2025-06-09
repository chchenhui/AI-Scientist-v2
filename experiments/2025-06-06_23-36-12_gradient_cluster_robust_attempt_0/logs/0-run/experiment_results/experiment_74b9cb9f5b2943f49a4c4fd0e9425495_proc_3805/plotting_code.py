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

for te_str, res_dict in experiment_data.get("train_epochs", {}).items():
    try:
        res = res_dict.get("synthetic", {})
        losses = res["losses"]
        metrics = res["metrics"]
        epochs = np.arange(len(losses["train"]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Synthetic Dataset (Hyperparam {te_str}) - Left: Loss Curves, Right: Worst-group Accuracy"
        )
        # Left: loss curves
        axes[0].plot(epochs, losses["train"], label="Train Loss")
        axes[0].plot(epochs, losses["val"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Train vs Validation Loss")
        axes[0].legend()
        # Right: worst-group accuracy curves
        axes[1].plot(epochs, metrics["train"], label="Train WG Acc")
        axes[1].plot(epochs, metrics["val"], label="Val WG Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Worst-group Accuracy")
        axes[1].set_title("Train vs Validation WG Accuracy")
        axes[1].legend()
        fname = f"synthetic_hparam_{te_str}_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for hyperparam {te_str}: {e}")
        plt.close("all")
