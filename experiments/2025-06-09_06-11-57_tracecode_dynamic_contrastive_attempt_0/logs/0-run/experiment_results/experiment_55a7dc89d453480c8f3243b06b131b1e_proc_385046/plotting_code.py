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

data = experiment_data["weight_decay_sweep"]["synthetic"]
wds = data["params"]
train_losses = data["losses"]["train"]
val_losses = data["losses"]["val"]
train_accs = data["metrics"]["train"]
val_accs = data["metrics"]["val"]
epochs = range(1, len(train_losses[0]) + 1)

# Plot 1: Loss curves
try:
    plt.figure()
    for wd, tloss, vloss in zip(wds, train_losses, val_losses):
        plt.plot(epochs, tloss, label=f"WD {wd} train")
        plt.plot(epochs, vloss, "--", label=f"WD {wd} val")
    plt.title("Synthetic dataset - Loss curves\nTrain (solid) vs Validation (dashed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: Accuracy curves
try:
    plt.figure()
    for wd, tac, vac in zip(wds, train_accs, val_accs):
        plt.plot(epochs, tac, label=f"WD {wd} train")
        plt.plot(epochs, vac, "--", label=f"WD {wd} val")
    plt.title(
        "Synthetic dataset - Accuracy curves\nTrain (solid) vs Validation (dashed)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# Plot 3: Final Validation Accuracy vs Weight Decay
try:
    plt.figure()
    final_vals = [accs[-1] for accs in val_accs]
    plt.bar([str(w) for w in wds], final_vals)
    plt.title("Synthetic dataset - Final Validation Accuracy vs Weight Decay")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final Validation Accuracy")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final val accuracy bar plot: {e}")
    plt.close()
