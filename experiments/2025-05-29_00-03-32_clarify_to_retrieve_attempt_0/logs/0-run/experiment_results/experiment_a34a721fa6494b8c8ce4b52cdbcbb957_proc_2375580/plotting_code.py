import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

ds = experiment_data.get("synthetic_xor", {})

# Plot loss curves
try:
    loss_train = ds["losses"]["train"]
    loss_val = ds["losses"]["val"]
    epochs = range(1, len(loss_train) + 1)
    plt.figure()
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves - synthetic_xor")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_synthetic_xor.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot CES curves
try:
    ces_train = ds["metrics"]["train"]
    ces_val = ds["metrics"]["val"]
    epochs = range(1, len(ces_train) + 1)
    plt.figure()
    plt.plot(epochs, ces_train, label="Train CES")
    plt.plot(epochs, ces_val, label="Val CES")
    plt.xlabel("Epoch")
    plt.ylabel("CES")
    plt.title("CES Curves - synthetic_xor")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "CES_curves_synthetic_xor.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES curves: {e}")
    plt.close()

# Plot class distribution on final validation epoch
try:
    preds = ds["predictions"][-1]
    gts = ds["ground_truth"][-1]
    counts_true = np.bincount(gts, minlength=2)
    counts_pred = np.bincount(preds, minlength=2)
    classes = np.arange(len(counts_true))
    width = 0.35
    plt.figure()
    plt.bar(classes - width / 2, counts_true, width, label="Ground Truth")
    plt.bar(classes + width / 2, counts_pred, width, label="Predictions")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(
        "Class Distribution (Left: Ground Truth, Right: Predictions) - synthetic_xor"
    )
    plt.xticks(classes)
    plt.legend()
    plt.savefig(os.path.join(working_dir, "class_distribution_synthetic_xor.png"))
    plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()

# Print final metrics summary
try:
    print(f"Final Train Loss: {loss_train[-1]:.4f}, Final Val Loss: {loss_val[-1]:.4f}")
    print(f"Final Train CES: {ces_train[-1]:.4f}, Final Val CES: {ces_val[-1]:.4f}")
except:
    pass
