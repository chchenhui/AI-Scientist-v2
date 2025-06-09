import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}
ed = experiment_data.get("alt_min_freq", {}).get("synthetic", {})

metrics_train = ed.get("metrics", {}).get("train", [])
metrics_val = ed.get("metrics", {}).get("val", [])
losses_train = ed.get("losses", {}).get("train", [])
losses_val = ed.get("losses", {}).get("val", [])
ratios = ed.get("ratios", [])

# Plot error curves
try:
    plt.figure()
    for (c, d), tr, vl in zip(ratios, metrics_train, metrics_val):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{c}:{d} train", linestyle="-")
        plt.plot(epochs, vl, label=f"{c}:{d} val", linestyle="--")
    plt.title("Synthetic Dataset: Training and Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.suptitle("Solid: Training, Dashed: Validation", fontsize=10)
    plt.savefig(os.path.join(working_dir, "synthetic_train_val_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves plot: {e}")
    plt.close()

# Plot loss curves
try:
    plt.figure()
    for (c, d), tr, vl in zip(ratios, losses_train, losses_val):
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, label=f"{c}:{d} train", linestyle="-")
        plt.plot(epochs, vl, label=f"{c}:{d} val", linestyle="--")
    plt.title("Synthetic Dataset: Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.suptitle("Solid: Training, Dashed: Validation", fontsize=10)
    plt.savefig(os.path.join(working_dir, "synthetic_train_val_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Print final validation errors
if ratios and metrics_val:
    print("Final validation errors per ratio:")
    for (c, d), vl in zip(ratios, metrics_val):
        print(f"Ratio {c}:{d} -> {vl[-1]:.4f}")
