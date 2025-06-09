import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# For each ablation, plot loss and accuracy curves
for ablation in ["random_negative", "hard_negative"]:
    # Loss curves
    try:
        plt.figure()
        for E, data in experiment_data[ablation]["synthetic"].items():
            epochs = list(range(1, len(data["losses"]["train"]) + 1))
            plt.plot(epochs, data["losses"]["train"], label=f"train E={E}")
            plt.plot(epochs, data["losses"]["val"], label=f"val E={E}")
        plt.title(f"{ablation}: Loss Curves (Synthetic)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"{ablation}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ablation}: {e}")
        plt.close()

    # Accuracy curves
    try:
        plt.figure()
        for E, data in experiment_data[ablation]["synthetic"].items():
            epochs = list(range(1, len(data["metrics"]["train"]) + 1))
            plt.plot(epochs, data["metrics"]["train"], label=f"train E={E}")
            plt.plot(epochs, data["metrics"]["val"], label=f"val E={E}")
        plt.title(f"{ablation}: Retrieval Accuracy vs Epoch (Synthetic)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        fname = f"{ablation}_accuracy_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ablation}: {e}")
        plt.close()
