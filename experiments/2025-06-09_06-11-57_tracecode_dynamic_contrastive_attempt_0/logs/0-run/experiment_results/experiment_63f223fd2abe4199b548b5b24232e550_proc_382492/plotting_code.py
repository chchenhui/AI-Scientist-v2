import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    data_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot loss curves
try:
    losses = experiment_data["trace_dataset"]["losses"]
    epochs = list(range(1, len(losses["train"]) + 1))
    plt.figure()
    plt.plot(epochs, losses["train"], label="Train Loss", color="blue")
    plt.plot(epochs, losses["val"], label="Val Loss", color="orange")
    plt.suptitle("Trace Dataset Loss Curves")
    plt.title("Left: Training Loss, Right: Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "trace_dataset_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot retrieval accuracy
try:
    metrics = experiment_data["trace_dataset"]["metrics"]
    epochs = list(range(1, len(metrics["train"]) + 1))
    plt.figure()
    plt.plot(epochs, metrics["train"], label="Train Acc", color="blue")
    plt.plot(epochs, metrics["val"], label="Val Acc", color="orange")
    plt.suptitle("Trace Dataset Retrieval Accuracy")
    plt.title("Left: Training Accuracy, Right: Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "trace_dataset_retrieval_accuracy.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
