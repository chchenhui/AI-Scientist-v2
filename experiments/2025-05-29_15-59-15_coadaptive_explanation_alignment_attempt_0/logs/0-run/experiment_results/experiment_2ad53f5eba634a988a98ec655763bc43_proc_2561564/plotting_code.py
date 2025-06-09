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

static = experiment_data.get("static_explainer", {})
metrics = static.get("metrics", {})
losses = static.get("losses", {})
preds = static.get("predictions", [])
gt = static.get("ground_truth", [])

# Accuracy curves
try:
    acc_train = metrics.get("train", [])
    acc_val = metrics.get("val", [])
    epochs = range(1, len(acc_train) + 1)
    plt.figure()
    plt.plot(epochs, acc_train, label="Train")
    plt.plot(epochs, acc_val, label="Val")
    plt.title("Static Explainer Accuracy Curves\nTraining vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "static_explainer_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Loss curves
try:
    loss_train = losses.get("train", [])
    loss_val = losses.get("val", [])
    epochs = range(1, len(loss_train) + 1)
    plt.figure()
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Val")
    plt.title("Static Explainer Loss Curves\nTraining vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "static_explainer_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Test set prediction distribution
try:
    if preds and gt:
        counts_pred = np.bincount(preds, minlength=2)
        counts_gt = np.bincount(gt, minlength=2)
        classes = np.arange(len(counts_gt))
        width = 0.35
        plt.figure()
        plt.bar(classes - width / 2, counts_gt, width, label="Ground Truth")
        plt.bar(classes + width / 2, counts_pred, width, label="Predictions")
        plt.title(
            "Static Explainer Test Predictions vs Ground Truth\nTest Set Class Distribution"
        )
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(classes)
        plt.legend()
        plt.savefig(os.path.join(working_dir, "static_explainer_test_distribution.png"))
        plt.close()
except Exception as e:
    print(f"Error creating prediction distribution plot: {e}")
    plt.close()
