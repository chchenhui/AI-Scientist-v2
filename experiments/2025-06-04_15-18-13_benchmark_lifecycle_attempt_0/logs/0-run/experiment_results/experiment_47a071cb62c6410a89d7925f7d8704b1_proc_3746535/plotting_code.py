import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# Plot training and validation losses
try:
    plt.figure()
    variants = data.get("batch_norm", {})
    for var, eps_dict in variants.items():
        for eps in [0.0, 0.05, 0.1, 0.2]:
            key = f"eps_{eps}"
            d = eps_dict.get(key, {})
            tr = d.get("losses", {}).get("train", [])
            vl = d.get("losses", {}).get("val", [])
            epochs = range(1, len(tr) + 1)
            plt.plot(epochs, tr, label=f"{var} ε={eps} train")
            plt.plot(epochs, vl, "--", label=f"{var} ε={eps} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST Training and Validation Loss Curves")
    plt.legend(loc="best")
    plt.savefig(os.path.join(working_dir, "MNIST_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot original test accuracy
try:
    plt.figure()
    for var, eps_dict in variants.items():
        for eps in [0.0, 0.05, 0.1, 0.2]:
            key = f"eps_{eps}"
            acc = eps_dict.get(key, {}).get("metrics", {}).get("orig_acc", [])
            epochs = range(1, len(acc) + 1)
            plt.plot(epochs, acc, label=f"{var} ε={eps}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST Original Test Accuracy Curves")
    plt.legend(loc="best")
    plt.savefig(os.path.join(working_dir, "MNIST_orig_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating original accuracy plot: {e}")
    plt.close()

# Plot augmented test accuracy
try:
    plt.figure()
    for var, eps_dict in variants.items():
        for eps in [0.0, 0.05, 0.1, 0.2]:
            key = f"eps_{eps}"
            acc = eps_dict.get(key, {}).get("metrics", {}).get("aug_acc", [])
            epochs = range(1, len(acc) + 1)
            plt.plot(epochs, acc, label=f"{var} ε={eps}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST Augmented Test Accuracy Curves")
    plt.legend(loc="best")
    plt.savefig(os.path.join(working_dir, "MNIST_aug_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating augmented accuracy plot: {e}")
    plt.close()
