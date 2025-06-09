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
    experiment_data = None

if experiment_data:
    data = experiment_data["atom_norm_projection"]["synthetic"]
    metrics_train = data["metrics"]["train"]
    metrics_val = data["metrics"]["val"]
    losses_train = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    betas = [0.5, 0.7, 0.9, 0.99]
    epochs = len(metrics_train[0])

    # Plot error curves
    try:
        plt.figure()
        for i, b in enumerate(betas):
            plt.plot(range(1, epochs + 1), metrics_train[i], label=f"Train β1={b}")
            plt.plot(range(1, epochs + 1), metrics_val[i], "--", label=f"Val β1={b}")
        plt.xlabel("Epoch")
        plt.ylabel("Relative Error")
        plt.title("Synthetic Dataset: Error Curves\nLeft: Solid=Train, Dashed=Val")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_error_curves_atom_norm_projection.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating error curves plot: {e}")
        plt.close()

    # Plot loss curves
    try:
        plt.figure()
        for i, b in enumerate(betas):
            plt.plot(range(1, epochs + 1), losses_train[i], label=f"Train β1={b}")
            plt.plot(range(1, epochs + 1), losses_val[i], "--", label=f"Val β1={b}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Synthetic Dataset: Loss Curves\nLeft: Solid=Train, Dashed=Val")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_loss_curves_atom_norm_projection.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # Print final validation errors
    for i, b in enumerate(betas):
        print(f"Final validation error for beta1={b}: {metrics_val[i][-1]:.4f}")
