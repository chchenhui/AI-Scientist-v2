import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp["weight_decay_variation"]["synthetic"]
    wds = data["weight_decays"]
    metrics_train = data["metrics"]["train"]
    metrics_val = data["metrics"]["val"]
    losses_train = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    preds = data["predictions"]
    truths = data["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    epochs = np.arange(metrics_train.shape[1])
    plt.figure()
    for i, wd in enumerate(wds):
        plt.plot(epochs, metrics_train[i], label=f"WD={wd:.0e} Train")
        plt.plot(epochs, metrics_val[i], "--", label=f"WD={wd:.0e} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Worst-Group Accuracy")
    plt.title("Worst-Group Accuracy across Epochs (Synthetic dataset)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating wg accuracy plot: {e}")
    plt.close()

try:
    epochs = np.arange(losses_train.shape[1])
    plt.figure()
    for i, wd in enumerate(wds):
        plt.plot(epochs, losses_train[i], label=f"WD={wd:.0e} Train")
        plt.plot(epochs, losses_val[i], "--", label=f"WD={wd:.0e} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss across Epochs (Synthetic dataset)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    test_acc = (preds == truths).mean(axis=1)
    labels = ["0" if wd == 0 else f"{wd:.0e}" for wd in wds]
    plt.figure()
    plt.bar(labels, test_acc, color="skyblue")
    plt.xlabel("Weight Decay")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy across Weight Decays (Synthetic dataset)")
    plt.savefig(os.path.join(working_dir, "synthetic_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy plot: {e}")
    plt.close()
