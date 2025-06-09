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

try:
    synthetic = experiment_data["group_inverse_frequency_reweighting"]["synthetic"]
    lrs = synthetic["lrs"]
    metrics = synthetic["metrics"]
    plt.figure()
    for i, lr in enumerate(lrs):
        plt.plot(metrics["train"][i], label=f"Train lr={lr}")
        plt.plot(metrics["val"][i], linestyle="--", label=f"Val lr={lr}")
    plt.title("Synthetic dataset: Worst‐Group Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Worst‐Group Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    losses = synthetic["losses"]
    plt.figure()
    for i, lr in enumerate(lrs):
        plt.plot(losses["train"][i], label=f"Train lr={lr}")
        plt.plot(losses["val"][i], linestyle="--", label=f"Val lr={lr}")
    plt.title("Synthetic dataset: Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()
