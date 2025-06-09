import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["hyperparam_tuning_type_1"]["synthetic_xor"]
    thresholds = data["thresholds"]
    losses_tr = np.array(data["losses"]["train"])  # shape: (n_thr, epochs)
    losses_val = np.array(data["losses"]["val"])
    ces_tr = np.array(data["metrics"]["train"])
    ces_val = np.array(data["metrics"]["val"])
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, thr in enumerate(thresholds):
        axes[0].plot(
            np.arange(1, losses_tr.shape[1] + 1), losses_tr[i], label=f"{thr:.3f}"
        )
        axes[1].plot(
            np.arange(1, losses_val.shape[1] + 1), losses_val[i], label=f"{thr:.3f}"
        )
    axes[0].set_title("Train Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[0].legend(title="Threshold")
    axes[1].legend(title="Threshold")
    fig.suptitle("Synthetic XOR Loss Curves\nLeft: Train Loss, Right: Validation Loss")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_xor_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot CES metric curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, thr in enumerate(thresholds):
        axes[0].plot(np.arange(1, ces_tr.shape[1] + 1), ces_tr[i], label=f"{thr:.3f}")
        axes[1].plot(np.arange(1, ces_val.shape[1] + 1), ces_val[i], label=f"{thr:.3f}")
    axes[0].set_title("Train CES Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CES")
    axes[1].set_title("Validation CES Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CES")
    axes[0].legend(title="Threshold")
    axes[1].legend(title="Threshold")
    fig.suptitle("Synthetic XOR CES Metrics\nLeft: Train CES, Right: Validation CES")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(working_dir, "synthetic_xor_CES_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES plot: {e}")
    plt.close()

# Print best threshold based on final validation CES
try:
    final_vals = ces_val[:, -1]
    idx = np.argmax(final_vals)
    print(
        f"Best threshold: {thresholds[idx]:.3f}, Final Validation CES: {final_vals[idx]:.4f}"
    )
except Exception as e:
    print(f"Error computing best threshold: {e}")
