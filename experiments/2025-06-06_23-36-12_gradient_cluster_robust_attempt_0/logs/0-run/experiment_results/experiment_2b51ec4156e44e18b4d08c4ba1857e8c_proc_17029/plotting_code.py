import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

sd = experiment_data.get("random_cluster_reweighting", {}).get("synthetic", {})
metrics_train = sd.get("metrics", {}).get("train")
metrics_val = sd.get("metrics", {}).get("val")
losses_train = sd.get("losses", {}).get("train")
losses_val = sd.get("losses", {}).get("val")

# Worst-group accuracy curves
try:
    epochs = np.arange(metrics_train.shape[1])
    num_runs = metrics_train.shape[0]
    run_labels = [f"run{i}" for i in range(num_runs)]
    plt.figure()
    plt.suptitle(
        "Worst-Group Accuracy Curves (Synthetic) - Left: Train, Right: Validation"
    )
    plt.subplot(1, 2, 1)
    for i in range(num_runs):
        plt.plot(epochs, metrics_train[i], label=run_labels[i])
    plt.title("Train Worst-Group Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    for i in range(num_runs):
        plt.plot(epochs, metrics_val[i], label=run_labels[i])
    plt.title("Val Worst-Group Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_worst_group_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating worst-group accuracy plot: {e}")
    plt.close()

# Loss curves
try:
    epochs = np.arange(losses_train.shape[1])
    num_runs = losses_train.shape[0]
    run_labels = [f"run{i}" for i in range(num_runs)]
    plt.figure()
    plt.suptitle("Loss Curves (Synthetic) - Left: Train, Right: Validation")
    plt.subplot(1, 2, 1)
    for i in range(num_runs):
        plt.plot(epochs, losses_train[i], label=run_labels[i])
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    for i in range(num_runs):
        plt.plot(epochs, losses_val[i], label=run_labels[i])
    plt.title("Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()
