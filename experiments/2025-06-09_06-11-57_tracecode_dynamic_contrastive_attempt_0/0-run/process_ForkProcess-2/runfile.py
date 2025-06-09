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

# Plot loss curves
try:
    losses = experiment_data["synthetic"]["losses"]
    epochs = np.arange(1, len(losses["train"]) + 1)
    plt.figure()
    plt.plot(epochs, losses["train"], label="Train Loss")
    plt.plot(epochs, losses["val"], label="Val Loss")
    plt.title("Loss Curves for synthetic dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    acc = experiment_data["synthetic"]["metrics"]
    epochs = np.arange(1, len(acc["train"]) + 1)
    plt.figure()
    plt.plot(epochs, acc["train"], label="Train Acc")
    plt.plot(epochs, acc["val"], label="Val Acc")
    plt.title("Accuracy Curves for synthetic dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Retrieval Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# Plot confusion matrix
try:
    preds = np.array(experiment_data["synthetic"]["predictions"])
    truths = np.array(experiment_data["synthetic"]["ground_truth"])
    num_classes = int(max(preds.max(), truths.max()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(truths, preds):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix for synthetic dataset")
    plt.xlabel("Predicted Group ID")
    plt.ylabel("True Group ID")
    plt.colorbar()
    plt.savefig(os.path.join(working_dir, "synthetic_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
