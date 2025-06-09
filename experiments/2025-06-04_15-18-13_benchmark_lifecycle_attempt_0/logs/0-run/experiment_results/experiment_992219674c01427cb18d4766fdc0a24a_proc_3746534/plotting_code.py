import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot train/val losses for ε=0.1 at each width
eps = 0.1
ekey = f"eps_{eps}"
for width in [8, 16, 32]:
    try:
        losses = data["width_ablation"][f"filters_{width}"][ekey]["losses"]
        epochs = np.arange(1, len(losses["train"]) + 1)
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.title(f"MNIST – Width={width}, ε={eps} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = f"mnist_loss_width_{width}_eps_{eps}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for width {width}: {e}")
        plt.close()

# Combined bar plots for final accuracies
try:
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    combos, orig_accs, aug_accs = [], [], []
    for width in [8, 16, 32]:
        for e in [0.0, 0.05, 0.1, 0.2]:
            key = data["width_ablation"][f"filters_{width}"][f"eps_{e}"]["metrics"]
            combos.append(f"W{width}_ε{e}")
            orig_accs.append(key["orig_acc"][-1])
            aug_accs.append(key["aug_acc"][-1])
    axs[0].bar(combos, orig_accs)
    axs[0].set_title("Left: Original Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].tick_params(axis="x", rotation=45)
    axs[1].bar(combos, aug_accs)
    axs[1].set_title("Right: Augmented Accuracy")
    axs[1].tick_params(axis="x", rotation=45)
    fig.suptitle("MNIST Final Test Accuracies Across Widths and ε")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mnist_final_accuracies.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy bar plots: {e}")
    plt.close()
