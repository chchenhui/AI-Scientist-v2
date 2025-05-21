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

# Plot combined accuracy and loss curves for each dataset
for ds in experiment_data.get("baseline", {}):
    try:
        epochs = np.arange(
            1, len(experiment_data["baseline"][ds]["metrics"]["train"]) + 1
        )
        plt.figure(figsize=(10, 5))
        # Accuracy subplot
        plt.subplot(1, 2, 1)
        for ab in experiment_data:
            tr = experiment_data[ab][ds]["metrics"]["train"]
            val = experiment_data[ab][ds]["metrics"]["val"]
            plt.plot(epochs, tr, marker="o", label=f"{ab} Train")
            plt.plot(epochs, val, marker="x", linestyle="--", label=f"{ab} Val")
        plt.title(f"{ds} Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        # Loss subplot
        plt.subplot(1, 2, 2)
        for ab in experiment_data:
            tr = experiment_data[ab][ds]["losses"]["train"]
            val = experiment_data[ab][ds]["losses"]["val"]
            plt.plot(epochs, tr, marker="o", label=f"{ab} Train")
            plt.plot(epochs, val, marker="x", linestyle="--", label=f"{ab} Val")
        plt.title(f"{ds} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.suptitle(f"Dataset: {ds} - Baseline vs Linear Probe")
        fname = os.path.join(working_dir, f"{ds}_acc_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot_{ds}_curves: {e}")
        plt.close()

# Plot bar chart of final validation accuracy by ablation and dataset
try:
    ds_list = list(experiment_data.get("baseline", {}))
    x = np.arange(len(ds_list))
    width = 0.35
    base_vals = [experiment_data["baseline"][d]["metrics"]["val"][-1] for d in ds_list]
    lin_vals = [
        experiment_data["linear_probe"][d]["metrics"]["val"][-1] for d in ds_list
    ]
    plt.figure()
    plt.bar(x - width / 2, base_vals, width, label="Baseline")
    plt.bar(x + width / 2, lin_vals, width, label="Linear Probe")
    plt.xticks(x, ds_list)
    plt.ylabel("Final Validation Accuracy")
    plt.title("Final Validation Accuracy by Dataset and Ablation")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "final_val_accuracy_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final_val_accuracy_bar plot: {e}")
    plt.close()
