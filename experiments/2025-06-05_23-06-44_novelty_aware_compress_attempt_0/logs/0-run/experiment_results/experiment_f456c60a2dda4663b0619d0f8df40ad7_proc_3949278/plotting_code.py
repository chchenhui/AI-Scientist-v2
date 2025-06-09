import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["hyperparam_tuning_type_1"]["synthetic"]
    chunk_sizes = data["chunk_sizes"]
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    train_metrics = data["metrics"]["train"]
    val_metrics = data["metrics"]["val"]
    preds_list = data["predictions"]
    gts_list = data["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot loss curves
try:
    plt.figure()
    epochs = range(1, len(train_losses[0]) + 1)
    for cs, tr, va in zip(chunk_sizes, train_losses, val_losses):
        plt.plot(epochs, tr, label=f"Train cs={cs}")
        plt.plot(epochs, va, linestyle="--", label=f"Val cs={cs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Synthetic Dataset - Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# Plot retention ratio curves
try:
    plt.figure()
    for cs, tr, va in zip(chunk_sizes, train_metrics, val_metrics):
        plt.plot(epochs, tr, label=f"Train Ret cs={cs}")
        plt.plot(epochs, va, linestyle="--", label=f"Val Ret cs={cs}")
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.title("Synthetic Dataset - Retention Ratio Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_retention_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating retention curve: {e}")
    plt.close()

# Plot sample predictions vs ground truth for largest chunk size
try:
    gt = gts_list[-1]
    preds = preds_list[-1]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(gt, marker="o")
    axs[0].set_title("Ground Truth")
    axs[0].set_xlabel("Time Step")
    axs[1].plot(preds, marker="o")
    axs[1].set_title("Generated Samples")
    axs[1].set_xlabel("Time Step")
    fig.suptitle("Synthetic Dataset - Sample Predictions")
    plt.savefig(os.path.join(working_dir, "synthetic_sample_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating sample predictions plot: {e}")
    plt.close()
