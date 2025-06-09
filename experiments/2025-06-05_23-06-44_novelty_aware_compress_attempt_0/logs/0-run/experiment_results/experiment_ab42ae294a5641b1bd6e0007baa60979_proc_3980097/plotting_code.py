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

for ds_key in experiment_data.get("baseline", {}):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"Metrics Curves for {ds_key}\n"
            "Left: Loss; Middle: Memory Retention Ratio; Right: Entropy-Weighted Memory Efficiency"
        )
        for ablation in ["baseline", "feedforward_identity"]:
            ed = experiment_data[ablation][ds_key]
            epochs = np.arange(1, len(ed["losses"]["train"]) + 1)
            # Loss curves
            axes[0].plot(epochs, ed["losses"]["train"], label=f"{ablation} Train")
            axes[0].plot(
                epochs, ed["losses"]["val"], linestyle="--", label=f"{ablation} Val"
            )
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            # Memory Retention Ratio curves
            mrr_tr = ed["metrics"]["Memory Retention Ratio"]["train"]
            mrr_val = ed["metrics"]["Memory Retention Ratio"]["val"]
            axes[1].plot(epochs, mrr_tr, label=f"{ablation} Train")
            axes[1].plot(epochs, mrr_val, linestyle="--", label=f"{ablation} Val")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Memory Retention Ratio")
            # Entropy-Weighted Memory Efficiency curves
            eme_tr = ed["metrics"]["Entropy-Weighted Memory Efficiency"]["train"]
            eme_val = ed["metrics"]["Entropy-Weighted Memory Efficiency"]["val"]
            axes[2].plot(epochs, eme_tr, label=f"{ablation} Train")
            axes[2].plot(epochs, eme_val, linestyle="--", label=f"{ablation} Val")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Entropy-Weighted Memory Efficiency")
        for ax in axes:
            ax.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"{ds_key}_metrics_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {ds_key}: {e}")
        plt.close()
