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

# Plot 1: Loss Curves
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model in zip(axs, experiment_data.keys()):
        disp = "LSTMEncoder" if model == "lstm" else "MeanPoolEncoder"
        for E in sorted(experiment_data[model]["synthetic"].keys()):
            losses = experiment_data[model]["synthetic"][E]["losses"]
            epochs = np.arange(1, len(losses["train"]) + 1)
            ax.plot(epochs, losses["train"], label=f"Train E={E}")
            ax.plot(epochs, losses["val"], "--", label=f"Val E={E}")
        ax.set_title(disp)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle(
        "Training and Validation Loss Curves (Synthetic)\nLeft: LSTMEncoder, Right: MeanPoolEncoder"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot 2: Retrieval Accuracy Curves
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model in zip(axs, experiment_data.keys()):
        disp = "LSTMEncoder" if model == "lstm" else "MeanPoolEncoder"
        for E in sorted(experiment_data[model]["synthetic"].keys()):
            metrics = experiment_data[model]["synthetic"][E]["metrics"]
            epochs = np.arange(1, len(metrics["train_acc"]) + 1)
            ax.plot(epochs, metrics["train_acc"], label=f"Train E={E}")
            ax.plot(epochs, metrics["val_acc"], "--", label=f"Val E={E}")
        ax.set_title(disp)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    fig.suptitle(
        "Retrieval Accuracy over Epochs (Synthetic)\nLeft: LSTMEncoder, Right: MeanPoolEncoder"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "synthetic_retrieval_accuracy.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# Plot 3: Cosine Similarity Gap Curves
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model in zip(axs, experiment_data.keys()):
        disp = "LSTMEncoder" if model == "lstm" else "MeanPoolEncoder"
        for E in sorted(experiment_data[model]["synthetic"].keys()):
            metrics = experiment_data[model]["synthetic"][E]["metrics"]
            epochs = np.arange(1, len(metrics["gap_train"]) + 1)
            ax.plot(epochs, metrics["gap_train"], label=f"Gap Train E={E}")
            ax.plot(epochs, metrics["gap_val"], "--", label=f"Gap Val E={E}")
        ax.set_title(disp)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Gap")
        ax.legend()
    fig.suptitle(
        "Cosine Similarity Gap over Epochs (Synthetic)\nLeft: LSTMEncoder, Right: MeanPoolEncoder"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "synthetic_gap_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating gap curves: {e}")
    plt.close()
