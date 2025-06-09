import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Reconstruct parameter lists
loss_names = ["mse", "mae", "huber"]
beta1_list = [0.5, 0.7, 0.9, 0.99]
syn = data["reconstruction_loss"]["synthetic"]
metrics = syn["metrics"]
losses_data = syn["losses"]

# Plot error curves
try:
    plt.figure()
    for i, loss_name in enumerate(loss_names):
        for j, b1 in enumerate(beta1_list):
            idx = i * len(beta1_list) + j
            tr = metrics["train"][idx]
            vl = metrics["val"][idx]
            plt.plot(tr, label=f"{loss_name}-train b1={b1}")
            plt.plot(vl, linestyle="--", label=f"{loss_name}-val b1={b1}")
    plt.title("Error Curves (Synthetic Dataset)")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves: {e}")
    plt.close()

# Plot reconstruction loss curves
try:
    plt.figure()
    for i, loss_name in enumerate(loss_names):
        for j, b1 in enumerate(beta1_list):
            idx = i * len(beta1_list) + j
            trl = losses_data["train"][idx]
            vll = losses_data["val"][idx]
            plt.plot(trl, label=f"{loss_name}-train b1={b1}")
            plt.plot(vll, linestyle="--", label=f"{loss_name}-val b1={b1}")
    plt.title("Loss Curves (Synthetic Dataset)")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Scatter plots of predictions vs ground truth for each loss type
for i, loss_name in enumerate(loss_names):
    idx = i * len(beta1_list)  # take the first beta1 setting for each loss
    try:
        plt.figure()
        preds = syn["predictions"][idx]
        gts = syn["ground_truth"][idx]
        p_flat = preds.flatten()
        g_flat = gts.flatten()
        plt.scatter(g_flat, p_flat, s=1)
        mn, mx = min(g_flat.min(), p_flat.min()), max(g_flat.max(), p_flat.max())
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.title(
            f"Scatter: Predictions vs Ground Truth\nDataset: Synthetic, Loss: {loss_name}"
        )
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.savefig(os.path.join(working_dir, f"synthetic_pred_vs_gt_{loss_name}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating pred vs gt for {loss_name}: {e}")
        plt.close()
