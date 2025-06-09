import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp_data["CNN_ENCODER_ABLATION"]["synthetic"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# Determine epoch settings
epochs = sorted(map(int, data.keys()))

# Plot loss curves
try:
    plt.figure(figsize=(8, 5))
    for E in epochs:
        d = data[E]
        x = range(1, len(d["losses"]["train"]) + 1)
        plt.plot(x, d["losses"]["train"], label=f"Train Loss E={E}")
        plt.plot(x, d["losses"]["val"], "--", label=f"Val Loss E={E}")
    plt.title("Synthetic Dataset Loss Curves for CNN Encoder Ablation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure(figsize=(8, 5))
    for E in epochs:
        d = data[E]
        x = range(1, len(d["metrics"]["train"]) + 1)
        plt.plot(x, d["metrics"]["train"], label=f"Train Acc E={E}")
        plt.plot(x, d["metrics"]["val"], "--", label=f"Val Acc E={E}")
    plt.title("Synthetic Dataset Accuracy Curves for CNN Encoder Ablation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# Plot final validation accuracy bar chart
try:
    final_acc = [data[E]["metrics"]["val"][-1] for E in epochs]
    plt.figure(figsize=(6, 4))
    plt.bar([str(E) for E in epochs], final_acc, color="skyblue")
    plt.title("Synthetic Dataset Final Validation Accuracy\nfor CNN Encoder Ablation")
    plt.xlabel("Epochs")
    plt.ylabel("Final Val Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final val accuracy plot: {e}")
    plt.close()
