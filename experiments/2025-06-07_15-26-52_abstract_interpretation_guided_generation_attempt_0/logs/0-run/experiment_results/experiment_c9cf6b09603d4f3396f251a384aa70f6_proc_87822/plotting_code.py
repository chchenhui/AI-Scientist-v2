import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ev = data["output_vocab_scaling"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ev = {}

# Loss curves
try:
    plt.figure()
    for key, v in ev.items():
        epochs = np.arange(1, len(v["losses"]["train"]) + 1)
        plt.plot(epochs, v["losses"]["train"], label=f"{key} train")
        plt.plot(epochs, v["losses"]["val"], "--", label=f"{key} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves for output_vocab_scaling")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "output_vocab_scaling_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# AICR curves
try:
    plt.figure()
    for key, v in ev.items():
        epochs = np.arange(1, len(v["metrics"]["AICR"]["train"]) + 1)
        plt.plot(epochs, v["metrics"]["AICR"]["train"], label=f"{key} train")
        plt.plot(epochs, v["metrics"]["AICR"]["val"], "--", label=f"{key} val")
    plt.xlabel("Epoch")
    plt.ylabel("AICR")
    plt.title("AICR Curves for output_vocab_scaling")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "output_vocab_scaling_AICR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AICR plot: {e}")
    plt.close()

# MeanIters curves
try:
    plt.figure()
    for key, v in ev.items():
        epochs = np.arange(1, len(v["metrics"]["MeanIters"]["train"]) + 1)
        plt.plot(epochs, v["metrics"]["MeanIters"]["train"], label=f"{key} train")
        plt.plot(epochs, v["metrics"]["MeanIters"]["val"], "--", label=f"{key} val")
    plt.xlabel("Epoch")
    plt.ylabel("MeanIters")
    plt.title("MeanIters Curves for output_vocab_scaling")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "output_vocab_scaling_MeanIters.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MeanIters plot: {e}")
    plt.close()
