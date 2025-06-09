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

# Plot retrieval accuracy
try:
    train_metrics = experiment_data["synthetic"]["metrics"]["train"]
    val_metrics = experiment_data["synthetic"]["metrics"]["val"]
    epochs = [m["epoch"] for m in train_metrics]
    acc_train = [m["retrieval_accuracy"] for m in train_metrics]
    acc_val = [m["retrieval_accuracy"] for m in val_metrics]

    plt.figure()
    plt.plot(epochs, acc_train, marker="o", label="Training")
    plt.plot(epochs, acc_val, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Retrieval Accuracy")
    plt.title("Retrieval Accuracy over Epochs on synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_retrieval_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating retrieval accuracy plot: {e}")
    plt.close()

# Plot loss curves
try:
    loss_train = experiment_data["synthetic"]["losses"]["train"]
    loss_val = experiment_data["synthetic"]["losses"]["val"]
    epochs = [m["epoch"] for m in experiment_data["synthetic"]["metrics"]["train"]]

    plt.figure()
    plt.plot(epochs, loss_train, marker="o", label="Training")
    plt.plot(epochs, loss_val, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs on synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()
