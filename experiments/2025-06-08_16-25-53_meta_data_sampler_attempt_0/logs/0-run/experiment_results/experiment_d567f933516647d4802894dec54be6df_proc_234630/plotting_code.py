import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

exp = experiment_data.get("ADAM_BETA2", {})
dataset = "synthetic"

# Plot 1: Loss curves
try:
    plt.figure()
    for key, val in exp.items():
        b2 = key.split("_", 1)[1]
        train = val[dataset]["losses"]["train"]
        val_loss = val[dataset]["losses"]["val"]
        epochs = range(len(train))
        plt.plot(epochs, train, label=f"beta2={b2} train")
        plt.plot(epochs, val_loss, linestyle="--", label=f"beta2={b2} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Training vs Validation Loss (Synthetic Dataset)\nTrain (solid) and Val (dashed)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()


# Spearman helper
def spearman_corr(a, b):
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return np.corrcoef(ar, br)[0, 1]


# Plot 2: Spearman correlation
try:
    plt.figure()
    for key, val in exp.items():
        b2 = key.split("_", 1)[1]
        preds = val[dataset]["predictions"]
        truths = val[dataset]["ground_truth"]
        corrs = [spearman_corr(p, t) for p, t in zip(preds, truths)]
        plt.plot(range(len(corrs)), corrs, marker="o", label=f"beta2={b2}")
        print(f"Final Spearman (beta2={b2}): {corrs[-1]:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Corr")
    plt.title("Spearman Correlation vs Epoch (Synthetic Dataset)\nHigher is better")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating spearman plot: {e}")
    plt.close()

# Plots 3–5: Scatter at final epoch
for key, val in exp.items():
    b2 = key.split("_", 1)[1]
    try:
        plt.figure()
        preds = np.array(val[dataset]["predictions"][-1])
        truths = np.array(val[dataset]["ground_truth"][-1])
        plt.scatter(truths, preds, alpha=0.6)
        plt.xlabel("Ground Truth")
        plt.ylabel("Predicted Meta-feedback")
        plt.title(
            f"Pred vs True Scatter (Synthetic Dataset, beta2={b2})\nScatter of model outputs"
        )
        plt.savefig(os.path.join(working_dir, f"synthetic_scatter_beta2_{b2}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter_beta2_{b2}: {e}")
        plt.close()
