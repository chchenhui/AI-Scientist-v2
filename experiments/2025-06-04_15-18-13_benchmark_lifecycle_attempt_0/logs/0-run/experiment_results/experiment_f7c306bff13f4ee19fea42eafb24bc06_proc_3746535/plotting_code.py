import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# load experiment_data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# plot loss curves
try:
    dropout_data = experiment_data["dropout_ablation"]
    p_keys = sorted(dropout_data.keys(), key=lambda x: float(x.split("_")[1]))
    fig, axes = plt.subplots(1, len(p_keys), figsize=(5 * len(p_keys), 4))
    for ax, p_key in zip(axes, p_keys):
        eps_keys = sorted(
            dropout_data[p_key].keys(), key=lambda x: float(x.split("_")[1])
        )
        p_val = float(p_key.split("_")[1])
        for eps_key in eps_keys:
            eps_val = float(eps_key.split("_")[1])
            losses = dropout_data[p_key][eps_key]["losses"]
            epochs = np.arange(1, len(losses["train"]) + 1)
            ax.plot(epochs, losses["train"], marker="o", label=f"train ε={eps_val}")
            ax.plot(epochs, losses["val"], marker="x", label=f"val   ε={eps_val}")
        ax.set_title(f"p={p_val}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("Loss Curves (MNIST)")
    plt.savefig(os.path.join(working_dir, "mnist_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# plot accuracy curves
try:
    dropout_data = experiment_data["dropout_ablation"]
    p_keys = sorted(dropout_data.keys(), key=lambda x: float(x.split("_")[1]))
    fig, axes = plt.subplots(1, len(p_keys), figsize=(5 * len(p_keys), 4))
    for ax, p_key in zip(axes, p_keys):
        eps_keys = sorted(
            dropout_data[p_key].keys(), key=lambda x: float(x.split("_")[1])
        )
        p_val = float(p_key.split("_")[1])
        for eps_key in eps_keys:
            eps_val = float(eps_key.split("_")[1])
            metrics = dropout_data[p_key][eps_key]["metrics"]
            epochs = np.arange(1, len(metrics["orig_acc"]) + 1)
            ax.plot(epochs, metrics["orig_acc"], marker="o", label=f"orig ε={eps_val}")
            ax.plot(epochs, metrics["aug_acc"], marker="x", label=f"aug  ε={eps_val}")
        ax.set_title(f"p={p_val}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    fig.suptitle("Accuracy Curves (MNIST) - Orig vs Aug")
    plt.savefig(os.path.join(working_dir, "mnist_accuracy_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()
