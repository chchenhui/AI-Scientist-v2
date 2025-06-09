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

# Sort learning rates
lr_strs = list(experiment_data.get("learning_rate", {}).keys())
lrs = sorted(lr_strs, key=lambda x: float(x))

# Plot loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lr in lrs:
        res = experiment_data["learning_rate"][lr]
        axes[0].plot(res["losses"]["train"], label=lr)
        axes[1].plot(res["losses"]["val"], label=lr)
    axes[0].set_title("Left: Training Loss - Code Dataset")
    axes[1].set_title("Right: Validation Loss - Code Dataset")
    fig.suptitle("Loss Curves per Learning Rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(title="LR")
    fig.savefig(os.path.join(working_dir, "code_dataset_loss_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close("all")

# Plot accuracy curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lr in lrs:
        res = experiment_data["learning_rate"][lr]
        axes[0].plot(res["metrics"]["train"], label=lr)
        axes[1].plot(res["metrics"]["val"], label=lr)
    axes[0].set_title("Left: Training Accuracy - Code Dataset")
    axes[1].set_title("Right: Validation Accuracy - Code Dataset")
    fig.suptitle("Accuracy Curves per Learning Rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(title="LR")
    fig.savefig(os.path.join(working_dir, "code_dataset_accuracy_curves.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close("all")

# Plot final validation accuracy bar chart
try:
    final_acc = [
        experiment_data["learning_rate"][lr]["metrics"]["val"][-1] for lr in lrs
    ]
    fig = plt.figure(figsize=(8, 5))
    plt.bar(lrs, final_acc, color="skyblue")
    plt.title("Final Validation Accuracy per Learning Rate - Code Dataset")
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(working_dir, "code_dataset_final_val_accuracy.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating final accuracy bar chart: {e}")
    plt.close("all")
