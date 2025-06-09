import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    syn = data["LOSS_BASED_SAMPLE_WEIGHTING"]["synthetic"]
    train_acc = syn["metrics"]["train"]
    val_acc = syn["metrics"]["val"]
    train_loss = syn["losses"]["train"]
    val_loss = syn["losses"]["val"]
    sample_weights = syn["sample_weights"]
    lrs = [1e-4, 1e-3, 1e-2]
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    plt.figure()
    epochs = np.arange(train_acc.shape[1])
    for i, lr in enumerate(lrs):
        plt.plot(epochs, train_acc[i], label=f"LR={lr} train")
        plt.plot(epochs, val_acc[i], "--", label=f"LR={lr} val")
    plt.xlabel("Epoch")
    plt.ylabel("Worst-Group Accuracy")
    plt.title(
        "Worst-Group Accuracy over Epochs (synthetic)\nTrain (solid) vs Validation (dashed)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure()
    for i, lr in enumerate(lrs):
        plt.plot(epochs, train_loss[i], label=f"LR={lr} train")
        plt.plot(epochs, val_loss[i], "--", label=f"LR={lr} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training/Validation Loss over Epochs (synthetic)\nCross-Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    fig, axes = plt.subplots(1, len(lrs), figsize=(5 * len(lrs), 4))
    fig.suptitle("Sample Weight Distribution (synthetic)")
    for i, lr in enumerate(lrs):
        ax = axes[i] if len(lrs) > 1 else axes
        ax.hist(sample_weights[i], bins=30)
        ax.set_title(f"LR={lr}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Frequency")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_sample_weights_hist.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
