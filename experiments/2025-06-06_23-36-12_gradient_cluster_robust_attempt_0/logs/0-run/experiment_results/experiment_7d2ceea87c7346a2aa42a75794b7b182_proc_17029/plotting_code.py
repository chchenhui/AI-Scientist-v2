import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = data["INPUT_FEATURE_CLUSTER_REWEIGHTING"]["synthetic"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    lrs = exp["lrs"]
    train_acc = exp["metrics"]["train"]
    val_acc = exp["metrics"]["val"]
    train_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    preds = exp["predictions"]
    gt = exp["ground_truth"]
    epochs = np.arange(train_acc.shape[1])

    try:
        plt.figure()
        plt.subplot(1, 2, 1)
        for i, lr in enumerate(lrs):
            plt.plot(epochs, train_acc[i], label=f"lr={lr}")
        plt.title("Left: Training WG Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("WG Accuracy")
        plt.legend()
        plt.subplot(1, 2, 2)
        for i, lr in enumerate(lrs):
            plt.plot(epochs, val_acc[i], label=f"lr={lr}")
        plt.title("Right: Validation WG Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("WG Accuracy")
        plt.legend()
        plt.suptitle("WG Accuracy Curves (synthetic dataset)")
        plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating wg accuracy plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.subplot(1, 2, 1)
        for i, lr in enumerate(lrs):
            plt.plot(epochs, train_loss[i], label=f"lr={lr}")
        plt.title("Left: Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        for i, lr in enumerate(lrs):
            plt.plot(epochs, val_loss[i], label=f"lr={lr}")
        plt.title("Right: Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.suptitle("Loss Curves (synthetic dataset)")
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Compute test accuracies
    test_acc = (preds == gt).mean(axis=1)
    print("Test accuracies per LR:")
    for lr, acc in zip(lrs, test_acc):
        print(f"LR={lr}: Test Accuracy={acc:.4f}")

    try:
        plt.figure()
        plt.bar([str(lr) for lr in lrs], test_acc)
        plt.title("Test Accuracy per Learning Rate (synthetic dataset)")
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar plot: {e}")
        plt.close()
