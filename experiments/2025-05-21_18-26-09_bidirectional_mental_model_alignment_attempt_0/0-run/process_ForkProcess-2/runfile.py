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
    experiment_data = None

if experiment_data:
    exp = experiment_data["synthetic"]
    epochs = exp["epochs"]
    losses_train = exp["losses"]["train"]
    losses_val = exp["losses"]["val"]
    align_train = exp["metrics"]["train"]
    align_val = exp["metrics"]["val"]
    preds = exp["predictions"]
    gts = exp["ground_truth"]

    # Plot loss curves
    try:
        plt.figure()
        plt.plot(epochs, losses_train, marker="o", label="Train Loss")
        plt.plot(epochs, losses_val, marker="s", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Synthetic Dataset Loss Curves\nTrain vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Plot alignment curves
    try:
        plt.figure()
        plt.plot(epochs, align_train, marker="o", label="Train Alignment")
        plt.plot(epochs, align_val, marker="s", label="Val Alignment")
        plt.xlabel("Epoch")
        plt.ylabel("Alignment (1 - JSD)")
        plt.title("Synthetic Dataset Alignment Metric\nTrain vs Validation Alignment")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating alignment plot: {e}")
        plt.close()

    # Compute and print final accuracy
    accuracy = np.mean(preds == gts)
    print(f"Final accuracy on synthetic dataset: {accuracy:.4f}")
