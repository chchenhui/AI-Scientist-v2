import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
beta1_list = [0.5, 0.7, 0.9, 0.99]

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for name, ed in experiment_data.get("synthetic_code_distribution", {}).items():
    metrics, losses = ed.get("metrics", {}), ed.get("losses", {})
    train_errs, val_errs = metrics.get("train", []), metrics.get("val", [])
    train_losses, val_losses = losses.get("train", []), losses.get("val", [])
    if not train_errs:
        continue
    epochs = len(train_errs[0])
    x = np.arange(1, epochs + 1)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, b1 in enumerate(beta1_list):
            axes[0].plot(x, train_errs[idx], label=f"beta1={b1}")
            axes[1].plot(x, val_errs[idx], label=f"beta1={b1}")
        axes[0].set_title("Training Error")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Relative Error")
        axes[1].set_title("Validation Error")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Relative Error")
        fig.suptitle(
            f"{name} Synthetic Distribution: Error Curves\nLeft: Training Error, Right: Validation Error"
        )
        axes[0].legend()
        axes[1].legend()
        plt.savefig(os.path.join(working_dir, f"{name}_error_curves.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for {name} error curves: {e}")
        plt.close()

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, b1 in enumerate(beta1_list):
            axes[0].plot(x, train_losses[idx], label=f"beta1={b1}")
            axes[1].plot(x, val_losses[idx], label=f"beta1={b1}")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        fig.suptitle(
            f"{name} Synthetic Distribution: Loss Curves\nLeft: Training Loss, Right: Validation Loss"
        )
        axes[0].legend()
        axes[1].legend()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curves.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating plot for {name} loss curves: {e}")
        plt.close()
