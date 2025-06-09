import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    datasets = exp_data.get("residual_connection_ablation", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    datasets = {}

# Plot loss curves
for ds, data in datasets.items():
    try:
        plt.figure()
        epochs = list(range(1, len(data["losses"]["train"]) + 1))
        plt.plot(epochs, data["losses"]["train"], label="Train Loss")
        plt.plot(epochs, data["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds} - Loss Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss_curve.png"))
    except Exception as e:
        print(f"Error creating loss curve for {ds}: {e}")
    finally:
        plt.close()

# Plot Memory Retention Ratio
for ds, data in datasets.items():
    try:
        plt.figure()
        epochs = list(
            range(1, len(data["metrics"]["Memory Retention Ratio"]["train"]) + 1)
        )
        plt.plot(
            epochs,
            data["metrics"]["Memory Retention Ratio"]["train"],
            label="Train MRR",
        )
        plt.plot(
            epochs, data["metrics"]["Memory Retention Ratio"]["val"], label="Val MRR"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Memory Retention Ratio")
        plt.title(f"{ds} - Memory Retention Ratio Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds}_memory_retention_ratio.png"))
    except Exception as e:
        print(f"Error creating Memory Retention Ratio for {ds}: {e}")
    finally:
        plt.close()

# Plot Entropy-Weighted Memory Efficiency
for ds, data in datasets.items():
    try:
        plt.figure()
        epochs = list(
            range(
                1,
                len(data["metrics"]["Entropy-Weighted Memory Efficiency"]["train"]) + 1,
            )
        )
        plt.plot(
            epochs,
            data["metrics"]["Entropy-Weighted Memory Efficiency"]["train"],
            label="Train EME",
        )
        plt.plot(
            epochs,
            data["metrics"]["Entropy-Weighted Memory Efficiency"]["val"],
            label="Val EME",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Entropy-Weighted Memory Efficiency")
        plt.title(f"{ds} - Entropy-Weighted Memory Efficiency Curve")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"{ds}_entropy_weighted_memory_efficiency.png")
        )
    except Exception as e:
        print(f"Error creating EME curve for {ds}: {e}")
    finally:
        plt.close()

# Plot predictions vs ground truth tokens
for ds, data in datasets.items():
    try:
        gt = data.get("ground_truth")
        preds = data.get("predictions")
        if gt is None or preds is None or len(gt) == 0:
            continue
        n = min(100, len(gt))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(gt[:n], color="blue")
        axes[0].set_title("Ground Truth Tokens")
        axes[1].plot(preds[:n], color="orange")
        axes[1].set_title("Predicted Tokens")
        fig.suptitle(f"{ds} - Left: Ground Truth, Right: Predicted Tokens")
        fig.savefig(os.path.join(working_dir, f"{ds}_tokens_comparison.png"))
    except Exception as e:
        print(f"Error creating token comparison for {ds}: {e}")
    finally:
        plt.close()
