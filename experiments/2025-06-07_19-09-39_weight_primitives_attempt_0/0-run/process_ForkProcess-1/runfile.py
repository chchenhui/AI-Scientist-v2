import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Extract synthetic dataset entries
synthetic = experiment_data.get("synthetic", {})
metrics = synthetic.get("metrics", {})
losses = synthetic.get("losses", {})
epochs = range(1, len(metrics.get("train", [])) + 1)

# Plot training and validation error
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train", []), label="Train Error")
    plt.plot(epochs, metrics.get("val", []), label="Val Error")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.title("Training vs Validation Error\nDataset: Synthetic")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_metrics_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot training and validation losses
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train Loss")
    plt.plot(epochs, losses.get("val", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss\nDataset: Synthetic")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_losses_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()
