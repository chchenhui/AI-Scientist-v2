import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["original_feature_removal"]["synthetic"]
    keys = sorted(data.keys())
    acc_train = {k: data[k]["metrics"]["train"] for k in keys}
    acc_val = {k: data[k]["metrics"]["val"] for k in keys}
    loss_train = {k: data[k]["losses"]["train"] for k in keys}
    loss_val = {k: data[k]["losses"]["val"] for k in keys}
except Exception as e:
    print(f"Error loading experiment data: {e}")
    keys = []

# Plot accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k in keys:
        axes[0].plot(acc_train[k], label=k)
        axes[1].plot(acc_val[k], label=k)
    axes[0].set_title("Training Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(loc="best")
    fig.suptitle("Accuracy Curves (Synthetic)\nLeft: Training, Right: Validation")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "synthetic_accuracy_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k in keys:
        axes[0].plot(loss_train[k], label=k)
        axes[1].plot(loss_val[k], label=k)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="best")
    fig.suptitle("Loss Curves (Synthetic)\nLeft: Training, Right: Validation")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Bar chart of final validation accuracy
try:
    final_val = [acc_val[k][-1] for k in keys]
    fig = plt.figure(figsize=(8, 5))
    plt.bar(keys, final_val)
    plt.xticks(rotation=45, ha="right")
    plt.title("Final Validation Accuracy (Synthetic)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    fig.savefig(os.path.join(working_dir, "synthetic_final_val_accuracy.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating final accuracy bar plot: {e}")
    plt.close()
