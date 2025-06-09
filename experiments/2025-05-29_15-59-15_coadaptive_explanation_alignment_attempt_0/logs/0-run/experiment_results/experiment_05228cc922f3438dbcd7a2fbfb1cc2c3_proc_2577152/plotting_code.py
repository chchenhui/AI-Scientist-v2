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

synth_data = experiment_data.get("network_depth_ablation", {}).get("synthetic", {})

for key, data in synth_data.items():
    preds = data["predictions"]
    gt = data["ground_truth"]
    acc = (preds == gt).mean()
    print(f"{key} test accuracy: {acc:.4f}")

for key, data in synth_data.items():
    train_acc = data["metrics"]["train"]
    val_acc = data["metrics"]["val"]
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(train_acc, label="Train Acc")
        axes[0].plot(val_acc, label="Val Acc")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Left: Accuracy over epochs")
        axes[0].legend()
        axes[1].plot(train_loss, label="Train Loss")
        axes[1].plot(val_loss, label="Val Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Right: Loss over epochs")
        axes[1].legend()
        fig.suptitle(f"Synthetic Dataset - {key}")
        plot_path = os.path.join(working_dir, f"synthetic_{key}_curves.png")
        fig.savefig(plot_path)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot_{key}: {e}")
        plt.close()
