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

schemes = experiment_data.get("weight_initialization", {}).keys()
data = experiment_data.get("weight_initialization", {})

# Plot training vs validation loss curves
try:
    plt.figure()
    for scheme in schemes:
        tr = data[scheme]["losses"]["train"]
        vl = data[scheme]["losses"]["val"]
        epochs = np.arange(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{scheme} Train")
        plt.plot(epochs, vl, linestyle="--", label=f"{scheme} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "MNIST Loss Curves (Train solid, Val dashed)\nWeight Initialization Schemes"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mnist_loss_curves_weight_init.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot original vs augmented accuracy curves
try:
    plt.figure()
    for scheme in schemes:
        orig = data[scheme]["metrics"]["orig_acc"]
        aug = data[scheme]["metrics"]["aug_acc"]
        epochs = np.arange(1, len(orig) + 1)
        plt.plot(epochs, orig, label=f"{scheme} Orig Acc")
        plt.plot(epochs, aug, linestyle="--", label=f"{scheme} Aug Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "MNIST Accuracy Curves (Original solid, Augmented dashed)\nWeight Initialization Schemes"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "mnist_accuracy_curves_weight_init.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()
