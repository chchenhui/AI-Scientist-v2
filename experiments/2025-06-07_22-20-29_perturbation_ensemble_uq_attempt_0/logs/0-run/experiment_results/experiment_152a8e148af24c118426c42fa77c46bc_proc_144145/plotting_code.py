import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
else:
    sd = data["adam_beta2"]["synthetic"]
    beta2_values = sd["beta2_values"]
    losses_tr = sd["losses"]["train"]
    losses_val = sd["losses"]["val"]
    auc_tr = sd["metrics"]["train"]
    auc_val = sd["metrics"]["val"]

    # Print final validation AUC per beta2
    print("Final validation AUC per beta2:")
    for b, aucs in zip(beta2_values, auc_val):
        print(f"beta2={b}: {aucs[-1]:.4f}")

    # Loss curves
    try:
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(
            "Loss Curves on synthetic dataset\nLeft: Training Loss, Right: Validation Loss"
        )
        ax1 = fig.add_subplot(1, 2, 1)
        for b, l in zip(beta2_values, losses_tr):
            ax1.plot(range(1, len(l) + 1), l, label=f"beta2={b}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax2 = fig.add_subplot(1, 2, 2)
        for b, l in zip(beta2_values, losses_val):
            ax2.plot(range(1, len(l) + 1), l, label=f"beta2={b}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # AUC curves
    try:
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(
            "AUC Curves on synthetic dataset\nLeft: Training AUC, Right: Validation AUC"
        )
        ax1 = fig.add_subplot(1, 2, 1)
        for b, a in zip(beta2_values, auc_tr):
            ax1.plot(range(1, len(a) + 1), a, label=f"beta2={b}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("AUC")
        ax1.legend()
        ax2 = fig.add_subplot(1, 2, 2)
        for b, a in zip(beta2_values, auc_val):
            ax2.plot(range(1, len(a) + 1), a, label=f"beta2={b}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC")
        ax2.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, "synthetic_auc_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating AUC plot: {e}")
        plt.close()
