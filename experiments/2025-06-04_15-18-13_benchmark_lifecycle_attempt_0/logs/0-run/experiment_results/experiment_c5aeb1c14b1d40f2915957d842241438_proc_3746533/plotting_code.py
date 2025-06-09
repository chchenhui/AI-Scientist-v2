import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    sched_data = experiment_data.get("lr_scheduler", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    sched_data = {}

# Loss curves comparison
try:
    plt.figure()
    for sched_name, info in sched_data.items():
        losses = info["losses"]
        epochs = range(1, len(losses["train"]) + 1)
        plt.plot(epochs, losses["train"], label=f"{sched_name} train")
        plt.plot(epochs, losses["val"], "--", label=f"{sched_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Scheduler Loss Curves Comparison (MNIST)\nTraining (solid), Validation (dashed)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "MNIST_loss_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss comparison plot: {e}")
    plt.close()

# Accuracy curves comparison
try:
    plt.figure()
    for sched_name, info in sched_data.items():
        metrics = info["metrics"]
        epochs = range(1, len(metrics["orig_acc"]) + 1)
        plt.plot(epochs, metrics["orig_acc"], label=f"{sched_name} orig_acc")
        plt.plot(epochs, metrics["aug_acc"], "--", label=f"{sched_name} aug_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Scheduler Accuracy Curves Comparison (MNIST)\nOriginal (solid), Augmented (dashed)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "MNIST_accuracy_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy comparison plot: {e}")
    plt.close()
