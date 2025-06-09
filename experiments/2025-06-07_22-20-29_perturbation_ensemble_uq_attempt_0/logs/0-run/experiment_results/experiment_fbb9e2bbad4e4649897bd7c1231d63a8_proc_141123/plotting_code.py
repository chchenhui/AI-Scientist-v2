import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp["synthetic"]
    loss_train = data["losses"]["train"]
    loss_val = data["losses"]["val"]
    auc_train = data["metrics"]["train"]
    auc_val = data["metrics"]["val"]
    divergences = np.array(data["predictions"])
    errors = np.array(data["ground_truth"])
except Exception as e:
    print(f"Error loading experiment data: {e}")
    loss_train, loss_val, auc_train, auc_val, divergences, errors = (
        [],
        [],
        [],
        [],
        np.array([]),
        np.array([]),
    )

# Plot training vs validation loss
try:
    plt.figure()
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.title(
        "Synthetic Dataset: Training and Validation Loss\nLeft: Train, Right: Val"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot training vs validation AUC
try:
    plt.figure()
    epochs = np.arange(1, len(auc_train) + 1)
    plt.plot(epochs, auc_train, label="Train AUC")
    plt.plot(epochs, auc_val, label="Val AUC")
    plt.title("Synthetic Dataset: Training and Validation AUC\nLeft: Train, Right: Val")
    plt.xlabel("Epoch")
    plt.ylabel("AUC-ROC")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_auc_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating AUC curve plot: {e}")
    plt.close()

# Plot histogram of divergences for correct vs incorrect
try:
    plt.figure()
    plt.hist(divergences[errors == 0], bins=20, alpha=0.6, label="Correct")
    plt.hist(divergences[errors == 1], bins=20, alpha=0.6, label="Incorrect")
    plt.title(
        "Synthetic Dataset: Divergence Distribution\nLeft: Correct, Right: Incorrect"
    )
    plt.xlabel("Uncertainty Score (Divergence)")
    plt.ylabel("Count")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_divergence_histogram.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating divergence histogram plot: {e}")
    plt.close()
