import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["n_clusters_tuning"]["synthetic"]
    n_clusters = ed["n_clusters"]
    train_acc = ed["metrics"]["train"]
    val_acc = ed["metrics"]["val"]
    train_loss = ed["losses"]["train"]
    val_loss = ed["losses"]["val"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for nc, t_acc in zip(n_clusters, train_acc):
        axes[0].plot(range(len(t_acc)), t_acc, label=f"n_clusters={nc}")
    for nc, v_acc in zip(n_clusters, val_acc):
        axes[1].plot(range(len(v_acc)), v_acc, label=f"n_clusters={nc}")
    fig.suptitle("Weighted Group Accuracy vs Epoch (Synthetic)")
    axes[0].set_title("Left: Training Accuracy")
    axes[1].set_title("Right: Validation Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("WG Accuracy")
        ax.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_wg_acc_vs_epoch.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close("all")

try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for nc, t_l in zip(n_clusters, train_loss):
        axes[0].plot(range(len(t_l)), t_l, label=f"n_clusters={nc}")
    for nc, v_l in zip(n_clusters, val_loss):
        axes[1].plot(range(len(v_l)), v_l, label=f"n_clusters={nc}")
    fig.suptitle("Cross-Entropy Loss vs Epoch (Synthetic)")
    axes[0].set_title("Left: Training Loss")
    axes[1].set_title("Right: Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_vs_epoch.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close("all")
