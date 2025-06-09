import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# derive ablations and datasets
ablations = list(data.keys())
datasets = list(data[ablations[0]].keys()) if ablations else []

for ds in datasets:
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Left: loss curves
        for abl in ablations:
            losses = data[abl][ds]["losses"]
            train = sorted(losses["train"], key=lambda x: x["epoch"])
            val = sorted(losses["val"], key=lambda x: x["epoch"])
            epochs = [e["epoch"] for e in train]
            axs[0].plot(epochs, [e["loss"] for e in train], label=f"train_{abl}")
            axs[0].plot(epochs, [e["loss"] for e in val], label=f"val_{abl}")
        axs[0].set_title("Loss curves (Left: Training & Validation)")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # Right: detection AUC
        for abl in ablations:
            det = sorted(
                data[abl][ds]["metrics"]["detection"], key=lambda x: x["epoch"]
            )
            epochs = [d["epoch"] for d in det]
            axs[1].plot(epochs, [d["auc_vote"] for d in det], "--", label=f"vote_{abl}")
            axs[1].plot(epochs, [d["auc_kl"] for d in det], "-.", label=f"kl_{abl}")
        axs[1].set_title("Detection AUC (KL & Vote)")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("AUC")
        axs[1].legend()

        fig.suptitle(f"Dataset: {ds}")
        fname = os.path.join(working_dir, f"{ds}_combined.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {ds}: {e}")
        plt.close()  # ensure closure even on error
