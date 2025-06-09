import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["synthetic_linear"]["metrics"]
    epochs = len(data["train_loss"])
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}
    epochs = 0

# Plot 1: Loss curves
try:
    plt.figure()
    plt.plot(range(epochs), data["train_loss"], label="Train Loss")
    plt.plot(range(epochs), data["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves (synthetic_linear)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_linear_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: Standard deviation before vs after rejuvenation
try:
    plt.figure()
    plt.plot(range(epochs), data["std_original"], label="Std Original")
    plt.plot(range(epochs), data["std_rejuvenated"], label="Std Rejuvenated")
    plt.xlabel("Epoch")
    plt.ylabel("Std of Model Agreement")
    plt.title("Model Agreement Std Over Epochs (synthetic_linear)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_linear_std_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating std curves plot: {e}")
    plt.close()

# Plot 3: Coefficient of Variation Ratio (CGR)
try:
    plt.figure()
    plt.plot(range(epochs), data["CGR"], label="CGR")
    plt.xlabel("Epoch")
    plt.ylabel("CGR")
    plt.title("Coefficient of Variation Ratio Over Epochs (synthetic_linear)")
    plt.savefig(os.path.join(working_dir, "synthetic_linear_CGR.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CGR plot: {e}")
    plt.close()

# Print final metric values if available
if epochs > 0:
    print(
        f"Final Metrics (Epoch {epochs-1}): "
        f"train_loss={data['train_loss'][-1]:.4f}, "
        f"val_loss={data['val_loss'][-1]:.4f}, "
        f"std_orig={data['std_original'][-1]:.4f}, "
        f"std_rej={data['std_rejuvenated'][-1]:.4f}, "
        f"CGR={data['CGR'][-1]:.4f}"
    )
