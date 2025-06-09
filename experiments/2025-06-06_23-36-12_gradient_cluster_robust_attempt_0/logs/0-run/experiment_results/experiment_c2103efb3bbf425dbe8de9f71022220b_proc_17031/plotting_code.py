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
    experiment_data = None

if experiment_data:
    sd = experiment_data["random_cluster_reweighting"]["synthetic"]
    train_metrics = sd["metrics"]["train"]
    val_metrics = sd["metrics"]["val"]
    train_losses = sd["losses"]["train"]
    val_losses = sd["losses"]["val"]
    preds = sd["predictions"]
    truths = sd["ground_truth"]
    # compute test accuracies
    test_acc = np.mean(preds == truths, axis=1)
    epochs = np.arange(train_metrics.shape[1])
    lr_list = [1e-4, 1e-3, 1e-2]

    try:
        plt.figure()
        for i, lr in enumerate(lr_list):
            plt.plot(epochs, train_metrics[i], label=f"train lr={lr}")
            plt.plot(epochs, val_metrics[i], "--", label=f"val lr={lr}")
        plt.title("Synthetic Dataset - Worst Group Accuracy vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Worst Group Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_worst_group_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating worst group accuracy plot: {e}")
        plt.close()

    try:
        plt.figure()
        for i, lr in enumerate(lr_list):
            plt.plot(epochs, train_losses[i], label=f"train lr={lr}")
            plt.plot(epochs, val_losses[i], "--", label=f"val lr={lr}")
        plt.title("Synthetic Dataset - Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.bar([str(lr) for lr in lr_list], test_acc)
        plt.title("Synthetic Dataset - Test Accuracy by Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Test Accuracy")
        plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar plot: {e}")
        plt.close()

    print(f"Test accuracies by learning rate: {dict(zip(lr_list, test_acc))}")
