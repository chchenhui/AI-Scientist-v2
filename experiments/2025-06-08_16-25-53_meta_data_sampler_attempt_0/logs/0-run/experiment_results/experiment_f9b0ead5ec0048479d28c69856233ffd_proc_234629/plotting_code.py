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
    experiment_data = {}


# Helper for Spearman correlation
def spearmanr(a, b):
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return np.corrcoef(ar, br)[0, 1]


# Plot 1: training & validation loss curves
try:
    plt.figure()
    hyp = experiment_data.get("hyperparam_tuning_type_1", {})
    for steps, cfg in hyp.items():
        m = cfg["synthetic"]["metrics"]
        epochs = np.arange(1, len(m["train"]) + 1)
        plt.plot(epochs, m["train"], label=f"{steps} steps train", linestyle="-")
        plt.plot(epochs, m["val"], label=f"{steps} steps val", linestyle="--")
    plt.title("Synthetic Dataset: Training & Validation Loss\nSolid=train, Dashed=val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Spearman correlation over epochs
try:
    plt.figure()
    for steps, cfg in hyp.items():
        preds = cfg["synthetic"]["predictions"]
        trues = cfg["synthetic"]["ground_truth"]
        cors = [spearmanr(p, t) for p, t in zip(preds, trues)]
        plt.plot(np.arange(1, len(cors) + 1), cors, label=f"{steps} steps")
    plt.title(
        "Synthetic Dataset: Spearman Correlation over Epochs\nDVN pred vs true contribution"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Spearman ρ")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_spearman_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: scatter predicted vs ground truth at final epoch for 10 steps
try:
    key = "10"
    cfg = hyp.get(key, {})
    p_final = np.array(cfg["synthetic"]["predictions"][-1])
    t_final = np.array(cfg["synthetic"]["ground_truth"][-1])
    plt.figure()
    plt.scatter(t_final, p_final, alpha=0.7)
    plt.title(
        "Synthetic Dataset: Pred vs GT Contributions (10 Steps)\nX: Ground Truth, Y: Predicted"
    )
    plt.xlabel("True Contribution")
    plt.ylabel("Predicted Contribution")
    plt.savefig(os.path.join(working_dir, "synthetic_scatter_steps_10.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
