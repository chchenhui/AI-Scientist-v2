import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    syn = exp["weight_decay"]["synthetic"]
    params = syn["params"]
    train_losses, val_losses = syn["losses"]["train"], syn["losses"]["val"]
    train_metrics, val_metrics = syn["metrics"]["train"], syn["metrics"]["val"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot 1: Loss curves
try:
    plt.figure()
    for w, tr, va in zip(params, train_losses, val_losses):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"Train wd={w}")
        plt.plot(epochs, va, "--", label=f"Val wd={w}")
    plt.title("Loss Curves - Synthetic Dataset (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: AICR curves
try:
    plt.figure()
    for w, tr, va in zip(params, train_metrics, val_metrics):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"Train AICR wd={w}")
        plt.plot(epochs, va, "--", label=f"Val AICR wd={w}")
    plt.title("AICR Curves - Synthetic Dataset (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("AICR")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_AICR_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating AICR curves plot: {e}")
    plt.close()

# Plot 3: Final validation AICR bar chart
try:
    final_val = [vm[-1] for vm in val_metrics]
    plt.figure()
    plt.bar([str(w) for w in params], final_val)
    plt.title("Final Validation AICR - Synthetic Dataset")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Val AICR")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_AICR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final AICR bar chart: {e}")
    plt.close()
