import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    synth = exp_data["lr_scheduler_gamma"]["synthetic"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    synth = {}

# Compute and print accuracies
for key, d in synth.items():
    preds = d.get("predictions", [])
    gts = d.get("ground_truth", [])
    if len(preds) and len(gts):
        acc = np.mean(preds == gts)
        print(f"{key} accuracy: {acc:.4f}")

# Plot Loss Curves
try:
    plt.figure()
    for key, d in synth.items():
        epochs = d["epochs"]
        train_losses = d["losses"]["train"]
        val_losses = d["losses"]["val"]
        plt.plot(epochs, train_losses, label=f"{key} Train")
        plt.plot(epochs, val_losses, "--", label=f"{key} Val")
    plt.title("Synthetic Dataset\nTrain vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot Alignment Curves
try:
    plt.figure()
    for key, d in synth.items():
        epochs = d["epochs"]
        train_align = d["metrics"]["train"]
        val_align = d["metrics"]["val"]
        plt.plot(epochs, train_align, label=f"{key} Train")
        plt.plot(epochs, val_align, "--", label=f"{key} Val")
    plt.title("Synthetic Dataset\nTrain vs Validation Alignment")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment plot: {e}")
    plt.close()
