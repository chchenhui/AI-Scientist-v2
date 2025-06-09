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

data_list = experiment_data.get("batch_size", {}).get("synthetic", [])

try:
    plt.figure()
    for entry in data_list:
        bs = entry["batch_size"]
        tl = entry["losses"]["train"]
        vl = entry["losses"]["val"]
        epochs = range(1, len(tl) + 1)
        plt.plot(epochs, tl, label=f"Train Loss bs={bs}")
        plt.plot(epochs, vl, label=f"Val Loss bs={bs}", linestyle="--")
    plt.title("Synthetic Dataset Loss Curves: Solid=Train, Dashed=Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure()
    for entry in data_list:
        bs = entry["batch_size"]
        ta = entry["metrics"]["train"]
        va = entry["metrics"]["val"]
        epochs = range(1, len(ta) + 1)
        plt.plot(epochs, ta, label=f"Train Acc bs={bs}")
        plt.plot(epochs, va, label=f"Val Acc bs={bs}", linestyle="--")
    plt.title("Synthetic Dataset Accuracy Curves: Solid=Train, Dashed=Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()
