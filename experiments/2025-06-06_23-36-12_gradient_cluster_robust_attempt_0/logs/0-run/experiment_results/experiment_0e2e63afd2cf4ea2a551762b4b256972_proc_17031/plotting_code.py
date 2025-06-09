import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    sd = experiment_data["NO_SPURIOUS_FEATURE"]["synthetic"]
    metrics_train = sd["metrics"]["train"]
    metrics_val = sd["metrics"]["val"]
    lrs = sd["lrs"]
    epochs = metrics_train.shape[1]
    plt.figure()
    for i, lr in enumerate(lrs):
        plt.plot(np.arange(epochs), metrics_train[i], label=f"Train lr={lr}")
        plt.plot(np.arange(epochs), metrics_val[i], "--", label=f"Val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Group Accuracy")
    plt.title(
        "Weighted Group Accuracy Curves (synthetic)\nTrain (solid) vs Validation (dashed)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating wg accuracy plot: {e}")
    plt.close()

try:
    sd = experiment_data["NO_SPURIOUS_FEATURE"]["synthetic"]
    losses_train = sd["losses"]["train"]
    losses_val = sd["losses"]["val"]
    lrs = sd["lrs"]
    epochs = losses_train.shape[1]
    plt.figure()
    for i, lr in enumerate(lrs):
        plt.plot(np.arange(epochs), losses_train[i], label=f"Train lr={lr}")
        plt.plot(np.arange(epochs), losses_val[i], "--", label=f"Val lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (synthetic)\nTrain (solid) vs Validation (dashed)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    sd = experiment_data["NO_SPURIOUS_FEATURE"]["synthetic"]
    preds = sd["predictions"]
    truths = sd["ground_truth"]
    lrs = sd["lrs"]
    accs = np.array([(preds[i] == truths).mean() for i in range(len(lrs))])
    plt.figure()
    plt.bar([str(lr) for lr in lrs], accs)
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy per Learning Rate (synthetic)\nTest set performance")
    plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
