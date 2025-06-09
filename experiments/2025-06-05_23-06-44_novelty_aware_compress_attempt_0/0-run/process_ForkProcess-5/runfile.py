import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
experiment_data = np.load(data_path, allow_pickle=True).item()

# Print final evaluation metrics for each learning rate
for lr, lr_data in experiment_data["learning_rate"].items():
    exp = lr_data["synthetic"]
    final_val_loss = exp["losses"]["val"][-1]
    final_val_ratio = exp["metrics"]["val"][-1]
    print(
        f"LR {lr} Final Val Loss: {final_val_loss:.4f}, Final Val Retention: {final_val_ratio:.4f}"
    )

# Loss curves plot
try:
    plt.figure()
    for lr, lr_data in experiment_data["learning_rate"].items():
        losses = lr_data["synthetic"]["losses"]
        epochs = range(len(losses["train"]))
        plt.plot(epochs, losses["train"], label=f"train lr={lr}")
        plt.plot(epochs, losses["val"], linestyle="--", label=f"val lr={lr}")
    plt.title("Synthetic Loss Curves Across Learning Rates\nDataset: synthetic")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Memory retention ratio plot
try:
    plt.figure()
    for lr, lr_data in experiment_data["learning_rate"].items():
        metrics = lr_data["synthetic"]["metrics"]
        epochs = range(len(metrics["train"]))
        plt.plot(epochs, metrics["train"], label=f"train lr={lr}")
        plt.plot(epochs, metrics["val"], linestyle="--", label=f"val lr={lr}")
    plt.title(
        "Synthetic Memory Retention Ratios Across Learning Rates\nDataset: synthetic"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_retention_ratios.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()
