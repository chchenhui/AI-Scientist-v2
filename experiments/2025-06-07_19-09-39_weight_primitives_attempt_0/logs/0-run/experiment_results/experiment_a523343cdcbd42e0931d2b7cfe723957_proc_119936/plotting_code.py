import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
try:
    edata = np.load(data_path, allow_pickle=True).item()
    syn = edata["data_dimensionality"]["synthetic"]
    dims = syn["dims"]
    errs_train = syn["metrics"]["train"]
    errs_val = syn["metrics"]["val"]
    loss_train = syn["losses"]["train"]
    loss_val = syn["losses"]["val"]
    gts = syn["ground_truth"]
    preds = syn["predictions"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot error curves
try:
    plt.figure()
    for i, d in enumerate(dims):
        epochs = len(errs_train[i])
        plt.plot(range(1, epochs + 1), errs_train[i], label=f"Train Err (dim={d})")
        plt.plot(range(1, epochs + 1), errs_val[i], "--", label=f"Val Err (dim={d})")
    plt.title("Training and Validation Relative Errors (synthetic)")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves: {e}")
    plt.close()

# Plot loss curves
try:
    plt.figure()
    for i, d in enumerate(dims):
        epochs = len(loss_train[i])
        plt.plot(range(1, epochs + 1), loss_train[i], label=f"Train Loss (dim={d})")
        plt.plot(range(1, epochs + 1), loss_val[i], "--", label=f"Val Loss (dim={d})")
    plt.title("Training and Validation Reconstruction Loss (synthetic)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Histograms for each dimension
for i, d in enumerate(dims):
    try:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(np.array(gts[i]).flatten(), bins=50)
        plt.title(f"Ground Truth (synthetic, dim={d})")
        plt.subplot(1, 2, 2)
        plt.hist(np.array(preds[i]).flatten(), bins=50)
        plt.title(f"Generated Samples (synthetic, dim={d})")
        plt.suptitle("Left: Ground Truth, Right: Generated Samples")
        fname = f"synthetic_histogram_dim_{d}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating histogram for dim={d}: {e}")
        plt.close()
