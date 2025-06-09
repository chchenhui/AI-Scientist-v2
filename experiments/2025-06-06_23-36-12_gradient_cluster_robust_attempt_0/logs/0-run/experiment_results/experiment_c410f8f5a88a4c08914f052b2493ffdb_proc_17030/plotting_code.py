import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    syn = data["SGD_OPTIMIZER"]["synthetic"]
    lrs = syn["learning_rates"]
    acc_tr = syn["metrics"]["train"]
    acc_val = syn["metrics"]["val"]
    loss_tr = syn["losses"]["train"]
    loss_val = syn["losses"]["val"]
    preds = syn["predictions"]
    truths = syn["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    plt.figure()
    for i, lr in enumerate(lrs):
        plt.plot(acc_tr[i], label=f"Train wg acc lr={lr}")
        plt.plot(acc_val[i], "--", label=f"Val wg acc lr={lr}")
    plt.title("Worst-Group Accuracy on synthetic dataset\nSolid: Train, Dashed: Val")
    plt.xlabel("Epoch")
    plt.ylabel("Worst-Group Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

try:
    plt.figure()
    for i, lr in enumerate(lrs):
        plt.plot(loss_tr[i], label=f"Train loss lr={lr}")
        plt.plot(loss_val[i], "--", label=f"Val loss lr={lr}")
    plt.title("Loss Curves on synthetic dataset\nSolid: Train, Dashed: Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    test_acc = [(preds[i] == truths).mean() for i in range(len(lrs))]
    plt.figure()
    plt.bar([str(lr) for lr in lrs], test_acc)
    plt.title(
        "Test Accuracy by Learning Rate on synthetic dataset\nBars: Test Accuracy"
    )
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy_by_lr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
    plt.close()
