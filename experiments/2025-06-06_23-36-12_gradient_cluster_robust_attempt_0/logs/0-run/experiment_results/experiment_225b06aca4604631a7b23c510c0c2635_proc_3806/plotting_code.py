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

ls_dict = experiment_data.get("label_smoothing", {})

# Plot worst-group accuracy curves
try:
    plt.figure()
    for eps, info in ls_dict.items():
        syn = info["synthetic"]
        train_acc = syn["metrics"]["train"]
        val_acc = syn["metrics"]["val"]
        epochs = np.arange(len(train_acc))
        plt.plot(epochs, train_acc, label=f"{eps} train")
        plt.plot(epochs, val_acc, "--", label=f"{eps} val")
    plt.title("Worst-Group Accuracy vs Epochs\nSynthetic dataset, label smoothing")
    plt.xlabel("Epoch")
    plt.ylabel("Worst-Group Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_worst_group_accuracy_vs_epochs.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating wg_acc plot: {e}")
    plt.close()

# Plot loss curves
try:
    plt.figure()
    for eps, info in ls_dict.items():
        syn = info["synthetic"]
        train_loss = syn["losses"]["train"]
        val_loss = syn["losses"]["val"]
        epochs = np.arange(len(train_loss))
        plt.plot(epochs, train_loss, label=f"{eps} train")
        plt.plot(epochs, val_loss, "--", label=f"{eps} val")
    plt.title("Loss vs Epochs\nSynthetic dataset, label smoothing")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_vs_epochs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Print final validation metrics
for eps, info in ls_dict.items():
    syn = info["synthetic"]
    final_val_acc = syn["metrics"]["val"][-1]
    final_val_loss = syn["losses"]["val"][-1]
    print(
        f"eps={eps} final val wg_acc={final_val_acc:.4f}, val loss={final_val_loss:.4f}"
    )
