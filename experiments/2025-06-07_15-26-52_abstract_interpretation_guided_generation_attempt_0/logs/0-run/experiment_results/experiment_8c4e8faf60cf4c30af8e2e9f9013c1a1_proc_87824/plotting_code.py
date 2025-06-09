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
else:
    data = experiment_data["fixed_random_embedding"]["synthetic"]
    lrs = data["params"]
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    epochs = range(1, len(train_losses[0]) + 1)
    # Plot loss curves
    try:
        plt.figure()
        for lr, tr, va in zip(lrs, train_losses, val_losses):
            plt.plot(epochs, tr, label=f"Train LR={lr}")
            plt.plot(epochs, va, "--", label=f"Val LR={lr}")
        plt.title("Loss Curves on synthetic dataset")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()
    # Plot generation success rates
    try:
        metrics_train = data["metrics"]["train"]
        metrics_val = data["metrics"]["val"]
        plt.figure()
        for lr, tr, va in zip(lrs, metrics_train, metrics_val):
            plt.plot(epochs, tr, label=f"Train SR LR={lr}")
            plt.plot(epochs, va, "--", label=f"Val SR LR={lr}")
        plt.title("Generation Success Rates on synthetic dataset")
        plt.xlabel("Epoch")
        plt.ylabel("Success Rate")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "synthetic_generation_success_rates.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating success rate plot: {e}")
        plt.close()
