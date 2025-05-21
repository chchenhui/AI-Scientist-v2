import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    exp_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp_data["label_smoothing"]["synthetic"]
    alphas = data["alphas"]
    epochs = data["epochs"]
    loss_train = data["losses"]["train"]
    loss_val = data["losses"]["val"]
    align_train = data["metrics"]["train"]
    align_val = data["metrics"]["val"]
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot loss curves
try:
    plt.figure()
    for i, alpha in enumerate(alphas):
        plt.plot(epochs, loss_train[i], label=f"α={alpha} Train")
        plt.plot(epochs, loss_val[i], "--", label=f"α={alpha} Val")
    plt.title("Synthetic Dataset: Loss Curves (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot alignment curves
try:
    plt.figure()
    for i, alpha in enumerate(alphas):
        plt.plot(epochs, align_train[i], label=f"α={alpha} Train")
        plt.plot(epochs, align_val[i], "--", label=f"α={alpha} Val")
    plt.title("Synthetic Dataset: Alignment Curves (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("Alignment (1−JSD)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_alignment_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating alignment curves: {e}")
    plt.close()

# Plot final validation alignment vs alpha
try:
    final_val_align = [lst[-1] for lst in align_val]
    plt.figure()
    plt.bar([str(a) for a in alphas], final_val_align)
    plt.title("Synthetic Dataset: Final Validation Alignment vs α")
    plt.xlabel("Label Smoothing α")
    plt.ylabel("Final Validation Alignment (1−JSD)")
    plt.savefig(os.path.join(working_dir, "synthetic_final_val_alignment.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final alignment bar chart: {e}")
    plt.close()

# Print out final metrics
print("Final validation alignment for each alpha:")
for alpha, val in zip(alphas, final_val_align):
    print(f"α={alpha}: {val:.4f}")
