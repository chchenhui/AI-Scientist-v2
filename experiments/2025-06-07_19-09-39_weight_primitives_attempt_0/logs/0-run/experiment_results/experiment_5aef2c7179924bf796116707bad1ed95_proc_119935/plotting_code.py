import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["lr_schedules"]["synthetic"]
    schedules = data["schedules"]
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    train_errs = data["metrics"]["train"]
    val_errs = data["metrics"]["val"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Loss curves
try:
    plt.figure()
    for sched, t_loss, v_loss in zip(schedules, train_losses, val_losses):
        plt.plot(range(1, len(t_loss) + 1), t_loss, label=f"{sched} train")
        plt.plot(range(1, len(v_loss) + 1), v_loss, "--", label=f"{sched} val")
    plt.title("Loss Curves for Synthetic Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Error curves
try:
    plt.figure()
    for sched, t_err, v_err in zip(schedules, train_errs, val_errs):
        plt.plot(range(1, len(t_err) + 1), t_err, label=f"{sched} train")
        plt.plot(range(1, len(v_err) + 1), "--", label=f"{sched} val")
    plt.title("Error Curves for Synthetic Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error curves plot: {e}")
    plt.close()

# Final validation error bar chart
try:
    final_val = [errs[-1] for errs in val_errs]
    plt.figure()
    plt.bar(schedules, final_val)
    plt.title("Final Validation Error Comparison for Synthetic Dataset")
    plt.xlabel("Learning-Rate Schedule")
    plt.ylabel("Final Validation Error")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_error.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final validation error plot: {e}")
    plt.close()
