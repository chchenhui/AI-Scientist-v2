import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# Plot loss curves
try:
    plt.figure()
    for abl in data:
        for ds in data[abl]:
            loss_tr = data[abl][ds]["losses"]["train"]
            loss_val = data[abl][ds]["losses"]["val"]
            epochs = range(1, len(loss_tr) + 1)
            plt.plot(epochs, loss_tr, label=f"{abl}-{ds} train")
            plt.plot(epochs, loss_val, "--", label=f"{abl}-{ds} val")
    plt.title("Loss Curves across Datasets")
    plt.suptitle("Training: solid, Validation: dashed | Dataset: Text Classification")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure()
    for abl in data:
        for ds in data[abl]:
            acc_tr = data[abl][ds]["metrics"]["train"]
            acc_val = data[abl][ds]["metrics"]["val"]
            epochs = range(1, len(acc_tr) + 1)
            plt.plot(epochs, acc_tr, label=f"{abl}-{ds} train")
            plt.plot(epochs, acc_val, "--", label=f"{abl}-{ds} val")
    plt.title("Accuracy Curves across Datasets")
    plt.suptitle("Training: solid, Validation: dashed | Dataset: Text Classification")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# Plot meta-learning dynamics
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for abl in data:
        for ds in data[abl]:
            corrs = data[abl][ds]["corrs"]
            nmeta = data[abl][ds]["N_meta_history"]
            steps = range(1, len(corrs) + 1)
            axes[0].plot(steps, corrs, label=f"{abl}-{ds}")
            axes[1].plot(steps, nmeta, label=f"{abl}-{ds}")
    axes[0].set_title("Spearman Corr History")
    axes[0].set_xlabel("Meta Update Step")
    axes[0].set_ylabel("Spearman ρ")
    axes[1].set_title("N_meta History")
    axes[1].set_xlabel("Meta Update Step")
    axes[1].set_ylabel("N_meta")
    fig.suptitle("Meta-learning Dynamics | Dataset: Text Classification")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "meta_dynamics.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating meta dynamics plot: {e}")
    plt.close()
