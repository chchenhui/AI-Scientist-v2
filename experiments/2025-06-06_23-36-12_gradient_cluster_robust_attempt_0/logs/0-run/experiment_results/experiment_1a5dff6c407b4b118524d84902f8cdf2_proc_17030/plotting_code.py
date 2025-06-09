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

for exp_name, exp_data in experiment_data.items():
    for dataset_name, data in exp_data.items():
        lrs = data["learning_rates"]
        metrics_train = data["metrics"]["train"]
        metrics_val = data["metrics"]["val"]
        losses_train = data["losses"]["train"]
        losses_val = data["losses"]["val"]
        epochs = np.arange(1, metrics_train.shape[1] + 1)

        # Worst‐group accuracy curves
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for i, lr in enumerate(lrs):
                axes[0].plot(epochs, metrics_train[i], label=f"lr={lr}")
                axes[1].plot(epochs, metrics_val[i], label=f"lr={lr}")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Worst-group accuracy")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Worst-group accuracy")
            fig.suptitle("Worst‐group Accuracy Curves")
            fig.text(
                0.5,
                0.01,
                f"Left: Train, Right: Validation | Dataset: {dataset_name}",
                ha="center",
            )
            axes[0].legend(), axes[1].legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_wg_accuracy_curves.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating wg accuracy curves: {e}")
            plt.close()

        # Loss curves
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for i, lr in enumerate(lrs):
                axes[0].plot(epochs, losses_train[i], label=f"lr={lr}")
                axes[1].plot(epochs, losses_val[i], label=f"lr={lr}")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            fig.suptitle("Loss Curves")
            fig.text(
                0.5,
                0.01,
                f"Left: Train, Right: Validation | Dataset: {dataset_name}",
                ha="center",
            )
            axes[0].legend(), axes[1].legend()
            plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curves: {e}")
            plt.close()

        # Final validation worst‐group accuracy bar chart
        try:
            plt.figure()
            final_val_wg = metrics_val[:, -1]
            plt.bar([str(lr) for lr in lrs], final_val_wg)
            plt.xlabel("Learning rate")
            plt.ylabel("Final worst-group accuracy")
            plt.title(f"Final Validation Worst‐group Accuracy\nDataset: {dataset_name}")
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_final_val_wg_accuracy.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating final val wg accuracy plot: {e}")
            plt.close()
