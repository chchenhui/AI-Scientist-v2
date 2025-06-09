import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()["adam_beta1"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# 1) Loss curves
try:
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    betas = list(exp.keys())
    epochs = range(1, len(exp[betas[0]]["MLP"]["losses"]["train"]) + 1)
    colors = ["C0", "C1", "C2"]
    for i, b in enumerate(betas):
        # MLP
        tr = exp[b]["MLP"]["losses"]["train"]
        val = exp[b]["MLP"]["losses"]["val"]
        axes[0].plot(
            epochs,
            tr,
            color=colors[i],
            linestyle="-",
            label=f"MLP β={b.split('_')[1]} train",
        )
        axes[0].plot(
            epochs,
            val,
            color=colors[i],
            linestyle="--",
            label=f"MLP β={b.split('_')[1]} val",
        )
        # CNN
        tr = exp[b]["CNN"]["losses"]["train"]
        val = exp[b]["CNN"]["losses"]["val"]
        axes[1].plot(
            epochs,
            tr,
            color=colors[i],
            linestyle="-",
            label=f"CNN β={b.split('_')[1]} train",
        )
        axes[1].plot(
            epochs,
            val,
            color=colors[i],
            linestyle="--",
            label=f"CNN β={b.split('_')[1]} val",
        )
    fig.suptitle("Training vs Validation Loss - MNIST")
    axes[0].set_title("MLP Loss (solid=train, dashed=val)")
    axes[1].set_title("CNN Loss (solid=train, dashed=val)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "loss_curves_MNIST.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2) Accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, b in enumerate(betas):
        oa_mlp = exp[b]["MLP"]["metrics"]["orig_acc"]
        aa_mlp = exp[b]["MLP"]["metrics"]["aug_acc"]
        oa_cnn = exp[b]["CNN"]["metrics"]["orig_acc"]
        aa_cnn = exp[b]["CNN"]["metrics"]["aug_acc"]
        axes[0].plot(
            epochs,
            oa_mlp,
            color=colors[i],
            linestyle="-",
            label=f"MLP β={b.split('_')[1]}",
        )
        axes[0].plot(
            epochs,
            oa_cnn,
            color=colors[i],
            linestyle="--",
            label=f"CNN β={b.split('_')[1]}",
        )
        axes[1].plot(
            epochs,
            aa_mlp,
            color=colors[i],
            linestyle="-",
            label=f"MLP β={b.split('_')[1]}",
        )
        axes[1].plot(
            epochs,
            aa_cnn,
            color=colors[i],
            linestyle="--",
            label=f"CNN β={b.split('_')[1]}",
        )
    fig.suptitle("Accuracy Curves - MNIST")
    axes[0].set_title("Left: Original Accuracy")
    axes[1].set_title("Right: Augmented Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "accuracy_curves_MNIST.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# 3) CGR curves
try:
    plt.figure(figsize=(6, 4))
    for i, b in enumerate(betas):
        cgr = exp[b]["CGR"]
        plt.plot(epochs, cgr, color=colors[i], label=f"β={b.split('_')[1]}")
    plt.title("CGR over epochs - MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("CGR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "CGR_curves_MNIST.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CGR curves plot: {e}")
    plt.close()
