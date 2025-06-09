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

d = experiment_data["classification_head_depth"]["synthetic"]
head_depths = d["head_depths"]

try:
    plt.figure()
    for idx, hd in enumerate(head_depths):
        epochs = range(1, len(d["losses"]["train"][idx]) + 1)
        plt.plot(epochs, d["losses"]["train"][idx], label=f"Train Loss Depth {hd}")
        plt.plot(epochs, d["losses"]["val"][idx], label=f"Val Loss Depth {hd}")
    plt.suptitle("Loss Curves: Synthetic Dataset")
    plt.title("Left: Train Loss, Right: Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    plt.figure()
    for idx, hd in enumerate(head_depths):
        epochs = range(1, len(d["metrics"]["train"][idx]) + 1)
        plt.plot(epochs, d["metrics"]["train"][idx], label=f"Train AICR Depth {hd}")
        plt.plot(epochs, d["metrics"]["val"][idx], label=f"Val AICR Depth {hd}")
    plt.suptitle("AICR Metrics: Synthetic Dataset")
    plt.title("Left: Train Rate, Right: Validation Rate")
    plt.xlabel("Epoch")
    plt.ylabel("AICR Rate")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_aicr_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    plt.figure()
    for idx, hd in enumerate(head_depths):
        epochs = range(1, len(d["classification_accuracy"]["train"][idx]) + 1)
        plt.plot(
            epochs,
            d["classification_accuracy"]["train"][idx],
            label=f"Train Acc Depth {hd}",
        )
        plt.plot(
            epochs,
            d["classification_accuracy"]["val"][idx],
            label=f"Val Acc Depth {hd}",
        )
    plt.suptitle("Classification Accuracy: Synthetic Dataset")
    plt.title("Left: Train Accuracy, Right: Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_classification_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
