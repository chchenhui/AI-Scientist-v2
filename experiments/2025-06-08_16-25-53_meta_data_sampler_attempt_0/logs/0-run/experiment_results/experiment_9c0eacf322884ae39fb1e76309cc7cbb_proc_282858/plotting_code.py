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
    experiment_data = {}

ablate_data = experiment_data.get("Ablate_Loss_Feature", {})
for dataset_name, info in ablate_data.items():
    losses = info.get("losses", {})
    metrics = info.get("metrics", {})

    # Plot loss curves
    try:
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        if train_loss and val_loss:
            plt.figure()
            epochs = range(1, len(train_loss) + 1)
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss Curves (Train vs Validation) - Dataset: {dataset_name}")
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset_name}_loss_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    # Plot accuracy curves
    try:
        train_acc = metrics.get("train", [])
        val_acc = metrics.get("val", [])
        if train_acc and val_acc:
            plt.figure()
            epochs = range(1, len(train_acc) + 1)
            plt.plot(epochs, train_acc, label="Train Accuracy")
            plt.plot(epochs, val_acc, label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"Accuracy Curves (Train vs Validation) - Dataset: {dataset_name}"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dataset_name}_accuracy_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dataset_name}: {e}")
        plt.close()
