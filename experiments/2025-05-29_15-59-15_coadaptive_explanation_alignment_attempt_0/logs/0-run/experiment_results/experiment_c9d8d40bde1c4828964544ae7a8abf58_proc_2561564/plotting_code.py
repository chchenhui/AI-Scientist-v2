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

# Print final validation metrics
try:
    val_losses = experiment_data["synthetic"]["losses"]["val"]
    val_align = experiment_data["synthetic"]["metrics"]["val"]
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final validation alignment: {val_align[-1]:.4f}")
except Exception:
    pass

# Plot loss curves
try:
    losses = experiment_data["synthetic"]["losses"]
    epochs = range(1, len(losses["train"]) + 1)
    plt.figure()
    plt.suptitle("Synthetic Dataset")
    plt.title("Loss Curves\nTrain vs Validation")
    plt.plot(epochs, losses["train"], "b-", label="Train Loss")
    plt.plot(epochs, losses["val"], "r--", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot alignment curves
try:
    metrics = experiment_data["synthetic"]["metrics"]
    epochs = range(1, len(metrics["train"]) + 1)
    plt.figure()
    plt.suptitle("Synthetic Dataset")
    plt.title("Alignment Metrics\nTrain vs Validation")
    plt.plot(epochs, metrics["train"], "b-", label="Train Alignment")
    plt.plot(epochs, metrics["val"], "r--", label="Val Alignment")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot confusion matrices for selected epochs
epochs_to_plot = [0, 2, 4, 6, 9]
preds_list = experiment_data["synthetic"]["predictions"]
gt_list = experiment_data["synthetic"]["ground_truth"]
for idx in epochs_to_plot:
    try:
        preds = preds_list[idx]
        gt = gt_list[idx]
        cm = np.zeros((2, 2), int)
        for t, p in zip(gt, preds):
            cm[t, p] += 1
        plt.figure()
        plt.suptitle("Synthetic Dataset")
        plt.title(
            f"Confusion Matrix Epoch {idx+1}\nRows: Ground Truth, Cols: Predictions"
        )
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="white")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        fname = f"synthetic_confusion_epoch{idx+1}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot_confusion_epoch{idx+1}: {e}")
        plt.close()
