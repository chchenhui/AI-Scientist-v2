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

# Plot for Euclidean triplet
try:
    ablation = "euclidean"
    synthetic = experiment_data[ablation]["synthetic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for E, info in synthetic.items():
        epochs = list(range(1, len(info["losses"]["train"]) + 1))
        axes[0].plot(epochs, info["losses"]["train"], label=f"Train E={E}")
        axes[0].plot(epochs, info["losses"]["val"], linestyle="--", label=f"Val E={E}")
        axes[1].plot(epochs, info["metrics"]["train"], label=f"Train E={E}")
        axes[1].plot(epochs, info["metrics"]["val"], linestyle="--", label=f"Val E={E}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Left: Loss Curves (synthetic)")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Right: Accuracy Curves (synthetic)")
    axes[1].legend()
    fig.suptitle("synthetic dataset – euclidean triplet")
    plt.savefig(os.path.join(working_dir, "synthetic_euclidean_loss_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating euclidean plot: {e}")
    plt.close()

# Plot for Cosine triplet
try:
    ablation = "cosine"
    synthetic = experiment_data[ablation]["synthetic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for E, info in synthetic.items():
        epochs = list(range(1, len(info["losses"]["train"]) + 1))
        axes[0].plot(epochs, info["losses"]["train"], label=f"Train E={E}")
        axes[0].plot(epochs, info["losses"]["val"], linestyle="--", label=f"Val E={E}")
        axes[1].plot(epochs, info["metrics"]["train"], label=f"Train E={E}")
        axes[1].plot(epochs, info["metrics"]["val"], linestyle="--", label=f"Val E={E}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Left: Loss Curves (synthetic)")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Right: Accuracy Curves (synthetic)")
    axes[1].legend()
    fig.suptitle("synthetic dataset – cosine triplet")
    plt.savefig(os.path.join(working_dir, "synthetic_cosine_loss_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cosine plot: {e}")
    plt.close()
