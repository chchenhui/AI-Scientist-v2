import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = data["weight_decay"]["synthetic"]
    metrics_train = exp["metrics"]["train"]  # shape (4, epochs)
    metrics_val = exp["metrics"]["val"]
    losses_train = exp["losses"]["train"]
    losses_val = exp["losses"]["val"]
    predictions = exp["predictions"]  # shape (4, n_test, dim)
    ground_truth = exp["ground_truth"]  # shape (n_test, dim)
    weight_decays = [1e-5, 1e-4, 1e-3, 1e-2]
    epochs = metrics_train.shape[1]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot 1: Reconstruction error vs epoch for each weight decay
try:
    plt.figure()
    for i, wd in enumerate(weight_decays):
        plt.plot(range(1, epochs + 1), metrics_train[i], label=f"Train wd={wd}")
        plt.plot(range(1, epochs + 1), metrics_val[i], "--", label=f"Val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Error")
    plt.title("Synthetic Dataset: Reconstruction Error vs Epoch")
    plt.suptitle("Train (solid) vs Val (dashed) across weight decays")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_metrics_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: MSE loss vs epoch for each weight decay
try:
    plt.figure()
    for i, wd in enumerate(weight_decays):
        plt.plot(range(1, epochs + 1), losses_train[i], label=f"Train wd={wd}")
        plt.plot(range(1, epochs + 1), losses_val[i], "--", label=f"Val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Synthetic Dataset: MSE Loss vs Epoch")
    plt.suptitle("Train (solid) vs Val (dashed) across weight decays")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Final reconstruction error vs weight decay
try:
    final_tr = metrics_train[:, -1]
    final_val = metrics_val[:, -1]
    plt.figure()
    plt.semilogx(weight_decays, final_tr, "-o", label="Train Error")
    plt.semilogx(weight_decays, final_val, "-s", label="Val Error")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Reconstruction Error")
    plt.title("Synthetic Dataset: Final Reconstruction Error vs Weight Decay")
    plt.suptitle("Train vs Val errors")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_final_error_vs_wd.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Final MSE loss vs weight decay
try:
    final_tr_loss = losses_train[:, -1]
    final_val_loss = losses_val[:, -1]
    plt.figure()
    plt.semilogx(weight_decays, final_tr_loss, "-o", label="Train Loss")
    plt.semilogx(weight_decays, final_val_loss, "-s", label="Val Loss")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final MSE Loss")
    plt.title("Synthetic Dataset: Final MSE Loss vs Weight Decay")
    plt.suptitle("Train vs Val losses")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_final_loss_vs_wd.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Sample reconstruction for first test sample (Left: GT, Right: Pred)
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ground_truth[0])
    axes[0].set_title("Left: Ground Truth")
    axes[1].plot(predictions[0][0])
    axes[1].set_title("Right: Predicted Sample (wd=1e-5)")
    fig.suptitle("Synthetic Dataset: Sample 0 Reconstruction")
    plt.savefig(
        os.path.join(working_dir, "synthetic_sample0_reconstruction_wd1e-05.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()
