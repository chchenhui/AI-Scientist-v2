import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# extract synthetic ablation
syn = experiment_data.get("projection_head_ablation", {}).get("synthetic", {})

try:
    plt.figure()
    for head, runs in syn.items():
        for epochs, d in runs.items():
            tr = d["losses"]["train"]
            va = d["losses"]["val"]
            x = np.arange(1, len(tr) + 1)
            plt.plot(x, tr, label=f"{head}_{epochs}_train")
            plt.plot(x, va, label=f"{head}_{epochs}_val")
    plt.title("Synthetic dataset: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    for head, runs in syn.items():
        for epochs, d in runs.items():
            tr = d["metrics"]["train"]
            va = d["metrics"]["val"]
            x = np.arange(1, len(tr) + 1)
            plt.plot(x, tr, label=f"{head}_{epochs}_train_acc")
            plt.plot(x, va, label=f"{head}_{epochs}_val_acc")
    plt.title("Synthetic dataset: Training vs Validation Retrieval Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
