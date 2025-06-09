import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

conf_data = experiment_data.get("confidence_filter", {})
keys = sorted(conf_data.keys(), key=lambda k: float(k.split("_")[-1]))
# determine epoch range
n_epochs = len(conf_data[keys[0]]["metrics"]["train"])
epochs = np.arange(1, n_epochs + 1)

# Plot accuracy curves
try:
    plt.figure()
    for key in keys:
        thr = key.split("_")[-1]
        train_acc = conf_data[key]["metrics"]["train"]
        val_acc = conf_data[key]["metrics"]["val"]
        plt.plot(epochs, train_acc, label=f"Train thr={thr}")
        plt.plot(epochs, val_acc, "--", label=f"Val thr={thr}")
    plt.title("Training vs Validation Accuracy\nDataset: Synthetic Classification")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "confidence_filter_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot loss curves
try:
    plt.figure()
    for key in keys:
        thr = key.split("_")[-1]
        train_loss = conf_data[key]["losses"]["train"]
        val_loss = conf_data[key]["losses"]["val"]
        plt.plot(epochs, train_loss, label=f"Train thr={thr}")
        plt.plot(epochs, val_loss, "--", label=f"Val thr={thr}")
    plt.title("Training vs Validation Loss\nDataset: Synthetic Classification")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "confidence_filter_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot test accuracy bar chart
try:
    thr_vals = []
    test_accs = []
    for key in keys:
        thr_vals.append(key.split("_")[-1])
        preds = conf_data[key]["predictions"]
        gt = conf_data[key]["ground_truth"]
        test_accs.append((preds == gt).mean())
    plt.figure()
    plt.bar(thr_vals, test_accs, color="skyblue")
    plt.title(
        "Test Accuracy per Confidence Threshold\nDataset: Synthetic Classification"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Test Accuracy")
    for i, v in enumerate(test_accs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.savefig(os.path.join(working_dir, "confidence_filter_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
