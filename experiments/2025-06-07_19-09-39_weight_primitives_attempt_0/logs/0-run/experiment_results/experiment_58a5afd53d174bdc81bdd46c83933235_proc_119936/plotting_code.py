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
    experiment_data = {}

data = experiment_data.get("test_time_solver_ablation", {}).get("synthetic", {})
solver_names = data.get("solver_names", [])
train_err = data.get("metrics", {}).get("train_err", np.array([]))
val_err = data.get("metrics", {}).get("val_err", {})
train_loss = data.get("losses", {}).get("train_loss", np.array([]))
val_loss = data.get("losses", {}).get("val_loss", {})

# Plot 1: Training Error Curve
try:
    plt.figure()
    plt.plot(train_err, label="Train Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Training Error Curve (Synthetic Dataset)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_training_error_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Training Loss Curve
try:
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Synthetic Dataset)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_training_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Validation Error for Solvers
try:
    plt.figure()
    for name in solver_names:
        if name in val_err:
            plt.plot(val_err[name], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Error")
    plt.title("Validation Error Curve - Solver Ablation (Synthetic Dataset)")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_validation_error_solver_ablation.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Validation Loss for Solvers
try:
    plt.figure()
    for name in solver_names:
        if name in val_loss:
            plt.plot(val_loss[name], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Curve - Solver Ablation (Synthetic Dataset)")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_validation_loss_solver_ablation.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()
