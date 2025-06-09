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

ed = experiment_data["optimizer_choice"]["synthetic"]
metrics = ed["metrics"]
losses = ed["losses"]
predictions = ed["predictions"]
ground_truth = ed["ground_truth"]
opts = ed["optimizers"]

# Convert to numpy arrays
metrics_train = np.array(metrics["train"])
metrics_val = np.array(metrics["val"])
losses_train = np.array(losses["train"])
losses_val = np.array(losses["val"])

# Print final errors for each optimizer
for i, opt in enumerate(opts):
    print(
        f"{opt}: final train error={metrics_train[i, -1]:.4f}, final val error={metrics_val[i, -1]:.4f}"
    )

# Plot training error curves
try:
    plt.figure()
    for i, opt in enumerate(opts):
        plt.plot(metrics_train[i], label=opt)
    plt.title("Synthetic Dataset: Training Error Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_training_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating train error plot: {e}")
    plt.close()

# Plot validation error curves
try:
    plt.figure()
    for i, opt in enumerate(opts):
        plt.plot(metrics_val[i], label=opt)
    plt.title("Synthetic Dataset: Validation Error Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_validation_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation error plot: {e}")
    plt.close()

# Plot training loss curves
try:
    plt.figure()
    for i, opt in enumerate(opts):
        plt.plot(losses_train[i], label=opt)
    plt.title("Synthetic Dataset: Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_training_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot validation loss curves
try:
    plt.figure()
    for i, opt in enumerate(opts):
        plt.plot(losses_val[i], label=opt)
    plt.title("Synthetic Dataset: Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_validation_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

# Plot ground truth vs reconstructed sample for best optimizer
try:
    best_idx = np.argmin(metrics_val[:, -1])
    best_opt = opts[best_idx]
    gt = ground_truth[best_idx][0]
    pred = predictions[best_idx][0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(gt)
    axs[0].set_title("Ground Truth")
    axs[1].plot(pred)
    axs[1].set_title("Generated Sample")
    fig.suptitle(
        f"Synthetic Dataset Sample Reconstruction ({best_opt})\nLeft: Ground Truth, Right: Generated Samples"
    )
    plt.savefig(os.path.join(working_dir, f"synthetic_reconstruction_{best_opt}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating reconstruction plot: {e}")
    plt.close()
