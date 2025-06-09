import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Extract synthetic results
data = experiment_data.get("representation_cluster_reweighting", {}).get(
    "synthetic", {}
)
metrics_train = data.get("metrics", {}).get("train", np.empty((0, 0)))
metrics_val = data.get("metrics", {}).get("val", np.empty((0, 0)))
losses_train = data.get("losses", {}).get("train", np.empty((0, 0)))
losses_val = data.get("losses", {}).get("val", np.empty((0, 0)))
preds = data.get("predictions", np.empty((0,)))
truth = data.get("ground_truth", np.empty((0,)))

# 1) Worst‐Group Accuracy Curves
try:
    plt.figure()
    epochs = np.arange(metrics_train.shape[1])
    for run in range(metrics_train.shape[0]):
        plt.plot(epochs, metrics_train[run], "-", label=f"Run {run+1} Train")
        plt.plot(epochs, metrics_val[run], "--", label=f"Run {run+1} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Worst‐Group Accuracy")
    plt.title("Synthetic Dataset: Train vs Val Worst‐Group Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating wg accuracy curve: {e}")
    plt.close()

# 2) Loss Curves
try:
    plt.figure()
    epochs = np.arange(losses_train.shape[1])
    for run in range(losses_train.shape[0]):
        plt.plot(epochs, losses_train[run], "-", label=f"Run {run+1} Train")
        plt.plot(epochs, losses_val[run], "--", label=f"Run {run+1} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Dataset: Train vs Val Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 3) Test Accuracy Bar Chart
try:
    if preds.size and truth.size:
        # preds is shape (runs, N), truth is (N,)
        acc_test = np.mean(preds == truth[None, :], axis=1)
        plt.figure()
        runs = np.arange(acc_test.size)
        plt.bar(runs, acc_test)
        plt.xticks(runs, [f"Run {i+1}" for i in runs])
        plt.xlabel("Run")
        plt.ylabel("Test Accuracy")
        plt.title("Synthetic Dataset: Test Accuracy by Run")
        plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy_bar.png"))
        plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar: {e}")
    plt.close()
