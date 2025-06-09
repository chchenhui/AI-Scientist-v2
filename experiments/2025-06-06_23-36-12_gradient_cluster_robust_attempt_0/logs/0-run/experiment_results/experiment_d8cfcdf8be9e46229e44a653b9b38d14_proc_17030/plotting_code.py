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

try:
    sd = experiment_data["NO_CLUSTER_REWEIGHTING"]["synthetic"]
    train_acc = sd["metrics"]["train"]
    val_acc = sd["metrics"]["val"]
    train_loss = sd["losses"]["train"]
    val_loss = sd["losses"]["val"]
    preds = sd["predictions"]
    gt = sd["ground_truth"]
    # Print test accuracies
    test_accs = (preds == gt).mean(axis=1)
    for i, acc in enumerate(test_accs):
        print(f"Run {i} Test Accuracy: {acc:.4f}")
    epochs = np.arange(train_acc.shape[1])
    runs = train_acc.shape[0]
    labels = [f"run{i}" for i in range(runs)]
    # Plot worst‐group accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(runs):
        axes[0].plot(epochs, train_acc[i], label=labels[i])
        axes[1].plot(epochs, val_acc[i], label=labels[i])
    axes[0].set_title("Left: Training Worst‐Group Accuracy")
    axes[1].set_title("Right: Validation Worst‐Group Accuracy")
    fig.suptitle("synthetic dataset - Worst‐Group Accuracy Curves")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(os.path.join(working_dir, "synthetic_wg_accuracy.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()
try:
    # Plot loss curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(runs):
        axes[0].plot(epochs, train_loss[i], label=labels[i])
        axes[1].plot(epochs, val_loss[i], label=labels[i])
    axes[0].set_title("Left: Training Loss")
    axes[1].set_title("Right: Validation Loss")
    fig.suptitle("synthetic dataset - Loss Curves")
    axes[0].legend()
    axes[1].legend()
    fig.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()
