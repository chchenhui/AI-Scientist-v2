import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Iterate over methods and datasets
for method_name, dsets in experiment_data.items():
    for dataset_name, exp in dsets.items():
        metrics = exp.get("metrics", {})
        losses = exp.get("losses", {})
        # Plot worst-group accuracy curves
        try:
            plt.figure()
            train_acc = metrics.get("train")
            val_acc = metrics.get("val")
            if train_acc is not None and val_acc is not None:
                epochs = np.arange(train_acc.shape[1])
                for i in range(train_acc.shape[0]):
                    plt.plot(epochs, train_acc[i], label=f"Run {i+1} Train")
                    plt.plot(epochs, val_acc[i], "--", label=f"Run {i+1} Val")
                plt.suptitle(
                    f"{method_name.replace('_',' ').title()} - {dataset_name.title()} Dataset"
                )
                plt.title("Left: Train, Right: Validation (Worst-Group Accuracy)")
                plt.xlabel("Epoch")
                plt.ylabel("Worst-Group Accuracy")
                plt.legend()
                fname = f"{method_name}_{dataset_name}_wg_accuracy_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating wg_accuracy plot for {method_name}: {e}")
            plt.close()
        # Plot loss curves
        try:
            plt.figure()
            train_loss = losses.get("train")
            val_loss = losses.get("val")
            if train_loss is not None and val_loss is not None:
                epochs = np.arange(train_loss.shape[1])
                for i in range(train_loss.shape[0]):
                    plt.plot(epochs, train_loss[i], label=f"Run {i+1} Train")
                    plt.plot(epochs, val_loss[i], "--", label=f"Run {i+1} Val")
                plt.suptitle(
                    f"{method_name.replace('_',' ').title()} - {dataset_name.title()} Dataset"
                )
                plt.title("Left: Train, Right: Validation (Average Loss)")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                fname = f"{method_name}_{dataset_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {method_name}: {e}")
            plt.close()
