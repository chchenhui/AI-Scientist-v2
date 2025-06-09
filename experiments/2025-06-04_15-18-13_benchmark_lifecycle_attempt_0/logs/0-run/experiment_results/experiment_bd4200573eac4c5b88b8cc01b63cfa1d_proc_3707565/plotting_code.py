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
    experiment_data = {}

# Print final original and augmented accuracy for each model & n_epochs
for n in sorted(experiment_data.get("n_epochs", {}), key=lambda x: int(x)):
    run = experiment_data["n_epochs"][n]
    for model in run["models"]:
        orig_acc = run["models"][model]["metrics"]["orig_acc"][-1]
        aug_acc = run["models"][model]["metrics"]["aug_acc"][-1]
        print(f"{model} (n_epochs={n}): orig_acc={orig_acc:.4f}, aug_acc={aug_acc:.4f}")

# Plot training vs val loss for each n_epochs (MLP & CNN)
for n in sorted(experiment_data.get("n_epochs", {}), key=lambda x: int(x)):
    run = experiment_data["n_epochs"][n]
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # MLP losses
        mlp = run["models"]["MLP"]["losses"]
        epochs_mlp = range(1, len(mlp["train"]) + 1)
        axes[0].plot(epochs_mlp, mlp["train"], label="Train Loss")
        axes[0].plot(epochs_mlp, mlp["val"], label="Val Loss")
        axes[0].set_title("Left: Training vs Val Loss (MLP)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        # CNN losses
        cnn = run["models"]["CNN"]["losses"]
        epochs_cnn = range(1, len(cnn["train"]) + 1)
        axes[1].plot(epochs_cnn, cnn["train"], label="Train Loss")
        axes[1].plot(epochs_cnn, cnn["val"], label="Val Loss")
        axes[1].set_title("Right: Training vs Val Loss (CNN)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        fig.suptitle(f"MNIST Loss Curves (n_epochs={n})")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fp = os.path.join(working_dir, f"mnist_loss_n_epochs_{n}.png")
        plt.savefig(fp)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating loss plot for n_epochs={n}: {e}")
        plt.close()

# Plot CGR vs epoch for all n_epochs
try:
    plt.figure()
    for n in sorted(experiment_data.get("n_epochs", {}), key=lambda x: int(x)):
        cgr = experiment_data["n_epochs"][n]["cgr"]
        plt.plot(range(1, len(cgr) + 1), cgr, marker="o", label=f"n_epochs={n}")
    plt.title("CGR vs Epoch for MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("CGR")
    plt.legend()
    fp = os.path.join(working_dir, "mnist_cgr.png")
    plt.savefig(fp)
    plt.close()
except Exception as e:
    print(f"Error creating CGR plot: {e}")
    plt.close()
