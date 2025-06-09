import matplotlib.pyplot as plt
import numpy as np
import os


def spearman_corr(a, b):
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    return np.corrcoef(a_rank, b_rank)[0, 1]


working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Plot training and validation loss curves
try:
    beta_data = experiment_data.get("ADAM_BETA1", {})
    plt.figure()
    for beta1, info in sorted(beta_data.items(), key=lambda x: float(x[0])):
        m = info["synthetic"]["metrics"]
        epochs = np.arange(len(m["train"]))
        plt.plot(epochs, m["train"], label=f"β1={beta1} Train")
        plt.plot(epochs, m["val"], "--", label=f"β1={beta1} Val")
    plt.title("Synthetic Dataset Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    path1 = os.path.join(working_dir, "synthetic_training_val_loss.png")
    plt.savefig(path1)
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot final‐epoch DVN predictions vs ground truth
cors = {}
try:
    beta_data = experiment_data.get("ADAM_BETA1", {})
    n = len(beta_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)
    for ax, (beta1, info) in zip(
        axes, sorted(beta_data.items(), key=lambda x: float(x[0]))
    ):
        preds = np.array(info["synthetic"]["predictions"][-1])
        truth = np.array(info["synthetic"]["ground_truth"][-1])
        corr = spearman_corr(preds, truth)
        cors[beta1] = corr
        ax.scatter(truth, preds, alpha=0.7)
        ax.set_title(f"β1={beta1}\nSpearman={corr:.2f}")
        ax.set_xlabel("True Influence")
        ax.set_ylabel("Predicted Influence")
    fig.suptitle("Synthetic Dataset DVN Influence Prediction (Final Epoch)")
    path2 = os.path.join(working_dir, "synthetic_dvn_scatter_final_epoch.png")
    plt.savefig(path2)
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Print final‐epoch Spearman correlations
for beta1, corr in sorted(cors.items(), key=lambda x: float(x[0])):
    print(f"Final epoch Spearman Corr for β1={beta1}: {corr:.4f}")
