import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = data["adam_beta2"]["synthetic"]
    betas = exp["beta2"]
    metrics = exp["metrics"]
    losses = exp["losses"]
    preds = exp["predictions"]
    gts = exp["ground_truth"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    epochs = metrics["train"].shape[1]
    plt.figure()
    plt.suptitle("Synthetic dataset: Training and Validation Error vs Epoch")
    plt.title("Left: Training Error, Right: Validation Error")
    for i, b in enumerate(betas):
        plt.plot(range(1, epochs + 1), metrics["train"][i], label=f"Train β₂={b}")
        plt.plot(range(1, epochs + 1), metrics["val"][i], "--", label=f"Val β₂={b}")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

try:
    epochs = losses["train"].shape[1]
    plt.figure()
    plt.suptitle("Synthetic dataset: Reconstruction Loss vs Epoch")
    plt.title("Left: Train Recon Loss, Right: Val Recon Loss")
    for i, b in enumerate(betas):
        plt.plot(range(1, epochs + 1), losses["train"][i], label=f"Train β₂={b}")
        plt.plot(range(1, epochs + 1), losses["val"][i], "--", label=f"Val β₂={b}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

try:
    final_errors = []
    for i in range(len(betas)):
        diff = preds[i] - gts[i]
        rel_errs = np.linalg.norm(diff, axis=1) / np.linalg.norm(gts[i], axis=1)
        final_errors.append(rel_errs)
    plt.figure()
    plt.suptitle("Synthetic dataset: Validation Error Distribution at Final Epoch")
    plt.title("Relative Errors per Test Sample for Each β₂")
    for i, b in enumerate(betas):
        plt.hist(final_errors[i], bins=10, alpha=0.5, label=f"β₂={b}")
    plt.xlabel("Relative Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_val_error_hist.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()
