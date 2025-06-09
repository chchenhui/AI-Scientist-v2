import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract synthetic ablation results
data = experiment_data.get("dropout_ablation", {}).get("synthetic", {})
dropouts = data.get("params", [])
train_losses = data.get("losses", {}).get("train", [])
val_losses = data.get("losses", {}).get("val", [])
train_rates = data.get("metrics", {}).get("train", [])
val_rates = data.get("metrics", {}).get("val", [])

# Plot loss curves
try:
    plt.figure()
    for dr, tloss, vloss in zip(dropouts, train_losses, val_losses):
        epochs = range(1, len(tloss) + 1)
        plt.plot(epochs, tloss, "-o", label=f"Train dr={dr}")
        plt.plot(epochs, vloss, "-x", label=f"Val dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves (Train vs Val) - Synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot generation pass rate curves
try:
    plt.figure()
    for dr, trate, vrate in zip(dropouts, train_rates, val_rates):
        epochs = range(1, len(trate) + 1)
        plt.plot(epochs, trate, "-o", label=f"Train dr={dr}")
        plt.plot(epochs, vrate, "-x", label=f"Val dr={dr}")
    plt.xlabel("Epoch")
    plt.ylabel("Pass Rate")
    plt.title("Generation Pass Rates (Train vs Val) - Synthetic dataset")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_generation_rates.png"))
    plt.close()
except Exception as e:
    print(f"Error creating generation rates plot: {e}")
    plt.close()

# Print final validation metrics
try:
    for dr, rates in zip(dropouts, val_rates):
        print(
            f"Synthetic dataset - dropout {dr}: final val pass rate = {rates[-1]:.4f}"
        )
except Exception as e:
    print(f"Error printing evaluation metrics: {e}")
