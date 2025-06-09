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

exp = experiment_data["weight_decay"]["synthetic"]
decays = exp["decays"]
train_losses = exp["losses"]["train"]
val_losses = exp["losses"]["val"]
train_metrics = exp["metrics"]["train"]
val_metrics = exp["metrics"]["val"]

epochs = np.arange(train_losses.shape[1])

try:
    plt.figure()
    for i, wd in enumerate(decays):
        plt.plot(epochs, train_losses[i], label=f"wd={wd} train")
        plt.plot(epochs, val_losses[i], linestyle="--", label=f"wd={wd} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Loss Curves\nTraining (solid) vs Validation (dashed) for synthetic dataset"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure()
    for i, wd in enumerate(decays):
        plt.plot(epochs, train_metrics[i], label=f"wd={wd} train")
        plt.plot(epochs, val_metrics[i], linestyle="--", label=f"wd={wd} val")
    plt.xlabel("Epoch")
    plt.ylabel("Worst-group Accuracy")
    plt.title(
        "Worst-group Accuracy Curves\nTraining (solid) vs Validation (dashed) for synthetic dataset"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wgacc_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    final_val = val_metrics[:, -1]
    plt.figure()
    plt.plot(decays, final_val, marker="o")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Validation Worst-group Accuracy")
    plt.title(
        "Final Validation Worst-group Accuracy vs Weight Decay\nsynthetic dataset"
    )
    plt.savefig(os.path.join(working_dir, "synthetic_final_wgacc_vs_wd.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
