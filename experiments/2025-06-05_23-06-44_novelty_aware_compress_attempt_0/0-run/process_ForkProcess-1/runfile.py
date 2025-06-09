import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

losses = experiment_data.get("synthetic", {}).get("losses", {})
metrics = experiment_data.get("synthetic", {}).get("metrics", {})
preds = experiment_data.get("synthetic", {}).get("predictions", [])
gt = experiment_data.get("synthetic", {}).get("ground_truth", [])

print("Final Train Loss:", losses.get("train", []))
print("Final Val Loss:", losses.get("val", []))
print("Final Train Metric:", metrics.get("train", []))
print("Final Val Metric:", metrics.get("val", []))

try:
    plt.figure()
    tl = losses.get("train", [])
    vl = losses.get("val", [])
    epochs = range(1, len(tl) + 1)
    plt.plot(epochs, tl, label="Train Loss")
    plt.plot(epochs, vl, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Dataset Loss Curves\nTraining vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

try:
    plt.figure()
    tm = metrics.get("train", [])
    vm = metrics.get("val", [])
    epochs = range(1, len(tm) + 1)
    plt.plot(epochs, tm, label="Train Ratio")
    plt.plot(epochs, vm, label="Val Ratio")
    plt.xlabel("Epoch")
    plt.ylabel("Memory Retention Ratio")
    plt.title("Synthetic Dataset Metric Curves\nMemory Retention over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_metric_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curve: {e}")
    plt.close()

try:
    if preds and gt:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(range(len(gt)), gt)
        axs[0].set_title("Ground Truth Sequence")
        axs[0].set_xlabel("Time Step")
        axs[0].set_ylabel("Token")
        axs[1].plot(range(len(preds)), preds, color="orange")
        axs[1].set_title("Generated Samples")
        axs[1].set_xlabel("Time Step")
        plt.suptitle(
            "Synthetic Dataset Generation\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.savefig(os.path.join(working_dir, "synthetic_generation_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating generation comparison: {e}")
    plt.close()
