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
    experiment_data = {}

data = experiment_data.get("synthetic", {})
losses = data.get("losses", {})
metrics = data.get("metrics", {})
epochs = list(range(1, len(losses.get("train", [])) + 1))

# Plot Loss Curve
try:
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train Loss")
    plt.plot(epochs, losses.get("val", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Dataset Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot Error-Free Generation Rate
try:
    plt.figure()
    plt.plot(epochs, metrics.get("train", []), label="Train Error-Free Rate")
    plt.plot(epochs, metrics.get("val", []), label="Val Error-Free Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Error-Free Rate")
    plt.title("Synthetic Dataset Error-Free Generation Rate")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_rate.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error rate plot: {e}")
    plt.close()
