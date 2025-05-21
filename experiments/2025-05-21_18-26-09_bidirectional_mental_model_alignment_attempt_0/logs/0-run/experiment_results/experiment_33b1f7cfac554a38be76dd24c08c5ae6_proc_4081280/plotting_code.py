import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Print best MAI per dataset and ablation
for ab in data:
    for ds in data[ab]:
        mai_list = data[ab][ds].get("mai", [])
        if mai_list:
            print(f"Dataset: {ds}, Ablation: {ab}, Best MAI: {max(mai_list):.4f}")

# Generate plots per dataset
datasets = next(iter(data.values())).keys()
for ds in datasets:
    # Loss curves
    try:
        plt.figure()
        for ab in data:
            losses = data[ab][ds]["losses"]
            epochs = np.arange(1, len(losses["train"]) + 1)
            plt.plot(epochs, losses["train"], label=f"{ab} train")
            plt.plot(epochs, losses["val"], label=f"{ab} val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {ds} Dataset")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

    # Alignment curves
    try:
        plt.figure()
        for ab in data:
            align = data[ab][ds]["alignments"]
            epochs = np.arange(1, len(align["train"]) + 1)
            plt.plot(epochs, align["train"], label=f"{ab} train")
            plt.plot(epochs, align["val"], label=f"{ab} val")
        plt.xlabel("Epoch")
        plt.ylabel("Alignment")
        plt.title(f"Alignment Curves for {ds} Dataset")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_alignment_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating alignment plot for {ds}: {e}")
        plt.close()

    # Validation MAI curves
    try:
        plt.figure()
        for ab in data:
            mai_list = data[ab][ds]["mai"]
            epochs = np.arange(1, len(mai_list) + 1)
            plt.plot(epochs, mai_list, label=f"{ab} val MAI")
        plt.xlabel("Epoch")
        plt.ylabel("MAI")
        plt.title(f"Validation MAI for {ds} Dataset")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_mai_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating MAI plot for {ds}: {e}")
        plt.close()
