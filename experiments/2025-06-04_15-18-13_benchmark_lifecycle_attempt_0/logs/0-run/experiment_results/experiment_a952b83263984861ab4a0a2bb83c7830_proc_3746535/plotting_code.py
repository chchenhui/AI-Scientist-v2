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
    experiment_data = {}

# Print final accuracies
opt_data = experiment_data.get("optimizer_choice", {})
print("Final Test Accuracies (Original / Augmented) by Optimizer:")
final_orig = {}
final_aug = {}
for opt, vals in opt_data.items():
    orig = vals["metrics"]["orig_acc"][-1] if vals["metrics"]["orig_acc"] else None
    aug = vals["metrics"]["aug_acc"][-1] if vals["metrics"]["aug_acc"] else None
    final_orig[opt] = orig
    final_aug[opt] = aug
    print(f"{opt}: {orig:.4f} / {aug:.4f}")

# Plot 1: Loss curves
try:
    plt.figure()
    epochs = range(1, len(next(iter(opt_data.values()))["losses"]["train"]) + 1)
    for opt, vals in opt_data.items():
        plt.plot(epochs, vals["losses"]["train"], label=f"{opt} Train")
        plt.plot(epochs, vals["losses"]["val"], "--", label=f"{opt} Val")
    plt.title("Loss Curves on MNIST (Training vs Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‐Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "loss_curves_mnist.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: Accuracy curves
try:
    plt.figure()
    epochs = range(1, len(next(iter(opt_data.values()))["metrics"]["orig_acc"]) + 1)
    for opt, vals in opt_data.items():
        plt.plot(epochs, vals["metrics"]["orig_acc"], label=f"{opt} Orig")
        plt.plot(epochs, vals["metrics"]["aug_acc"], "--", label=f"{opt} Aug")
    plt.title("Test Accuracy on MNIST (Original vs Augmented)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "accuracy_curves_mnist.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# Plot 3: Final accuracy bar chart
try:
    plt.figure()
    opts = list(final_orig.keys())
    x = np.arange(len(opts))
    width = 0.35
    orig_vals = [final_orig[o] for o in opts]
    aug_vals = [final_aug[o] for o in opts]
    plt.bar(x - width / 2, orig_vals, width, label="Original")
    plt.bar(x + width / 2, aug_vals, width, label="Augmented")
    plt.title("Final Test Accuracies on MNIST by Optimizer")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy")
    plt.xticks(x, opts)
    plt.legend()
    plt.savefig(os.path.join(working_dir, "final_accuracy_bar_mnist.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar chart: {e}")
    plt.close()
