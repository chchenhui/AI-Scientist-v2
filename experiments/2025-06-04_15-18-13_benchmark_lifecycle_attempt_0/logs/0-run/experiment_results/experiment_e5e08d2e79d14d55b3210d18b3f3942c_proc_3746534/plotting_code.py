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

try:
    mixup = experiment_data["mixup"]
    alphas = sorted([float(k.split("_")[1]) for k in mixup.keys()])
    epochs = list(range(1, len(next(iter(mixup.values()))["losses"]["train"]) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for alpha in alphas:
        key = f"alpha_{alpha}"
        plt.plot(epochs, mixup[key]["losses"]["train"], label=f"α={alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    for alpha in alphas:
        key = f"alpha_{alpha}"
        plt.plot(epochs, mixup[key]["losses"]["val"], label=f"α={alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.suptitle("MNIST Loss Curves\nLeft: Training Loss, Right: Validation Loss")
    plt.savefig(os.path.join(working_dir, "mnist_loss_curves_mixup.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

try:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for alpha in alphas:
        key = f"alpha_{alpha}"
        plt.plot(epochs, mixup[key]["metrics"]["orig_acc"], label=f"α={alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Original Test Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    for alpha in alphas:
        key = f"alpha_{alpha}"
        plt.plot(epochs, mixup[key]["metrics"]["aug_acc"], label=f"α={alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Augmented Test Accuracy")
    plt.suptitle(
        "MNIST Accuracy Curves\nLeft: Orig Test Acc, Right: Augmented Test Acc"
    )
    plt.savefig(os.path.join(working_dir, "mnist_accuracy_curves_mixup.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()
