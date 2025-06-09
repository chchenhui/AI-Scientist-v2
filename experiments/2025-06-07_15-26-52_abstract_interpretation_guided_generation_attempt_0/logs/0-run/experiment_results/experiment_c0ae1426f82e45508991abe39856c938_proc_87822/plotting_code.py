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
else:
    ds = data["label_smoothing"]["synthetic"]
    factors = ds["params"]
    train_losses = ds["losses"]["train"]
    val_losses = ds["losses"]["val"]
    train_rates = ds["metrics"]["train"]
    val_rates = ds["metrics"]["val"]
    train_iters = ds["mean_iterations"]["train"]
    val_iters = ds["mean_iterations"]["val"]
    epochs = range(1, len(train_losses[0]) + 1)

    # Plot losses
    try:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for l, s in zip(train_losses, factors):
            axs[0].plot(epochs, l, label=f"smooth={s}")
        for l, s in zip(val_losses, factors):
            axs[1].plot(epochs, l, label=f"smooth={s}")
        axs[0].set_title("Left: Training Loss")
        axs[1].set_title("Right: Validation Loss")
        axs[0].set_xlabel("Epoch")
        axs[1].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        fig.suptitle("Synthetic Dataset Loss vs Epoch - Label Smoothing Ablation")
        plt.savefig(os.path.join(working_dir, "synthetic_loss_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Plot pass rates
    try:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for r, s in zip(train_rates, factors):
            axs[0].plot(epochs, r, label=f"smooth={s}")
        for r, s in zip(val_rates, factors):
            axs[1].plot(epochs, r, label=f"smooth={s}")
        axs[0].set_title("Left: Training Pass Rate")
        axs[1].set_title("Right: Validation Pass Rate")
        axs[0].set_xlabel("Epoch")
        axs[1].set_xlabel("Epoch")
        axs[0].set_ylabel("Pass Rate")
        axs[1].set_ylabel("Pass Rate")
        axs[1].legend()
        fig.suptitle("Synthetic Dataset Pass Rate vs Epoch - Label Smoothing Ablation")
        plt.savefig(os.path.join(working_dir, "synthetic_pass_rate_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating pass rate plot: {e}")
        plt.close()

    # Plot mean iterations
    try:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for it, s in zip(train_iters, factors):
            axs[0].plot(epochs, it, label=f"smooth={s}")
        for it, s in zip(val_iters, factors):
            axs[1].plot(epochs, it, label=f"smooth={s}")
        axs[0].set_title("Left: Training Mean Iterations")
        axs[1].set_title("Right: Validation Mean Iterations")
        axs[0].set_xlabel("Epoch")
        axs[1].set_xlabel("Epoch")
        axs[0].set_ylabel("Mean Iters")
        axs[1].set_ylabel("Mean Iters")
        axs[1].legend()
        fig.suptitle(
            "Synthetic Dataset Mean Iterations vs Epoch - Label Smoothing Ablation"
        )
        plt.savefig(os.path.join(working_dir, "synthetic_mean_iterations_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating mean iterations plot: {e}")
        plt.close()

    # Print summary of final validation performance
    print("Summary of final validation performance:")
    for s, losses, rates in zip(factors, val_losses, val_rates):
        print(
            f"Label smoothing {s}: Final Val Loss={losses[-1]:.4f}, Final Val Pass Rate={rates[-1]:.4f}"
        )
