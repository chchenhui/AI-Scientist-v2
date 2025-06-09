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

# Extract arrays
loss_train = data.get("original", {}).get("losses", {}).get("train", [])
loss_val = data.get("original", {}).get("losses", {}).get("val", [])
orig_acc = data.get("original", {}).get("metrics", {}).get("orig_acc", [])
aug_acc = data.get("original", {}).get("metrics", {}).get("aug_acc", [])
cgr_vals = data.get("CGR", [])

# Determine epochs and split per model (MLP then CNN)
n_models = 2
n_epochs = len(loss_train) // n_models if n_models else 0
epochs = np.arange(1, n_epochs + 1)
train_mlp = loss_train[0::2]
train_cnn = loss_train[1::2]
val_mlp = loss_val[0::2]
val_cnn = loss_val[1::2]
orig_mlp = orig_acc[0::2]
orig_cnn = orig_acc[1::2]
aug_mlp = aug_acc[0::2]
aug_cnn = aug_acc[1::2]

# Plot loss curves
try:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_mlp, label="Train Loss")
    plt.plot(epochs, val_mlp, label="Val Loss")
    plt.title("Left: MLP on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_cnn, label="Train Loss")
    plt.plot(epochs, val_cnn, label="Val Loss")
    plt.title("Right: CNN on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.suptitle("Training and Validation Losses (MNIST)")
    plt.savefig(os.path.join(working_dir, "mnist_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs, orig_mlp, label="Orig Acc")
    plt.plot(epochs, aug_mlp, label="Aug Acc")
    plt.title("Left: MLP on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, orig_cnn, label="Orig Acc")
    plt.plot(epochs, aug_cnn, label="Aug Acc")
    plt.title("Right: CNN on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.suptitle("Original and Augmented Accuracy (MNIST)")
    plt.savefig(os.path.join(working_dir, "mnist_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# Plot CGR curve
try:
    plt.figure()
    plt.plot(np.arange(1, len(cgr_vals) + 1), cgr_vals, marker="o")
    plt.title("CGR across Epochs on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("CGR")
    plt.savefig(os.path.join(working_dir, "mnist_cgr_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CGR plot: {e}")
    plt.close()
