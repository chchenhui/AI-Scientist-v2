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
    experiment_data = {}

for name, data in experiment_data.items():
    synthetic = data.get("synthetic", {})
    # Loss curves
    try:
        plt.figure()
        for E, d in synthetic.items():
            epochs = range(1, len(d["losses"]["train"]) + 1)
            plt.plot(epochs, d["losses"]["train"], label=f"Train E={E}")
            plt.plot(epochs, d["losses"]["val"], label=f"Val E={E}")
        plt.title(f"{name} on synthetic: Loss curves")
        plt.suptitle("Left: Train Loss, Right: Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curves_synthetic.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {name}: {e}")
        plt.close()
    # Retrieval accuracy
    try:
        plt.figure()
        for E, d in synthetic.items():
            epochs = range(1, len(d["metrics"]["train"]) + 1)
            plt.plot(epochs, d["metrics"]["train"], label=f"Train E={E}")
            plt.plot(epochs, d["metrics"]["val"], label=f"Val E={E}")
        plt.title(f"{name} on synthetic: Retrieval Accuracy")
        plt.suptitle("Left: Train Accuracy, Right: Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_accuracy_synthetic.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {name}: {e}")
        plt.close()
    # Alignment gap
    try:
        plt.figure()
        for E, d in synthetic.items():
            epochs = range(1, len(d["metrics"]["align_gap_train"]) + 1)
            plt.plot(epochs, d["metrics"]["align_gap_train"], label=f"Train E={E}")
            plt.plot(epochs, d["metrics"]["align_gap_val"], label=f"Val E={E}")
        plt.title(f"{name} on synthetic: Alignment Gap")
        plt.suptitle("Left: Train Align Gap, Right: Val Align Gap")
        plt.xlabel("Epoch")
        plt.ylabel("Alignment Gap")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_align_gap_synthetic.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating align gap plot for {name}: {e}")
        plt.close()
