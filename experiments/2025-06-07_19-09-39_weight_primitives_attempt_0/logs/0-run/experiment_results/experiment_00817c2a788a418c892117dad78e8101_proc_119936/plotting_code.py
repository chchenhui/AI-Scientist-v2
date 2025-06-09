import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["sparsity_strength_ablation"]["synthetic"]
    train_errs = data["metrics"]["train"]
    val_errs = data["metrics"]["val"]
    train_losses = data["losses"]["train"]
    val_losses = data["losses"]["val"]
    sparsities = data["sparsity"]
    dict_errors = data["dict_error"]
    lambda_list = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    epochs = len(train_errs[0])
    x = np.arange(1, epochs + 1)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot error curves
try:
    plt.figure()
    for i, lam in enumerate(lambda_list):
        plt.plot(x, train_errs[i], label=f"train λ={lam}")
        plt.plot(x, val_errs[i], "--", label=f"val λ={lam}")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Error")
    plt.title("Synthetic Dataset: Training (solid) and Validation (dashed) Errors")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_error_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating error plot: {e}")
    plt.close()

# Plot loss curves
try:
    plt.figure()
    for i, lam in enumerate(lambda_list):
        plt.plot(x, train_losses[i], label=f"train λ={lam}")
        plt.plot(x, val_losses[i], "--", label=f"val λ={lam}")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Synthetic Dataset: Training (solid) and Validation (dashed) Losses")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot sparsity over epochs
try:
    plt.figure()
    for i, lam in enumerate(lambda_list):
        plt.plot(x, sparsities[i], label=f"λ={lam}")
    plt.xlabel("Epoch")
    plt.ylabel("Code Sparsity Fraction")
    plt.title("Synthetic Dataset: Code Sparsity Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_sparsity.png"))
    plt.close()
except Exception as e:
    print(f"Error creating sparsity plot: {e}")
    plt.close()

# Plot dictionary recovery error
try:
    plt.figure()
    for i, lam in enumerate(lambda_list):
        plt.plot(x, dict_errors[i], label=f"λ={lam}")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Dictionary Recovery Error")
    plt.title("Synthetic Dataset: Dictionary Recovery Error Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dict_recovery_error.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dict error plot: {e}")
    plt.close()

# Print final validation errors
if "lambda_list" in locals():
    print("Final Validation Errors per Lambda:")
    for lam, vals in zip(lambda_list, val_errs):
        print(f"Lambda {lam}: {vals[-1]:.4f}")
