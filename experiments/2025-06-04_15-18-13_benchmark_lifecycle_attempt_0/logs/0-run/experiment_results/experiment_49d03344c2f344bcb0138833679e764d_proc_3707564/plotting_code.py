import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_keys = sorted(experiment_data["learning_rate"].keys(), key=float)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot loss curves for MLP and CNN
for model in ["MLP", "CNN"]:
    try:
        fig = plt.figure(figsize=(10, 4))
        # Training loss
        ax1 = fig.add_subplot(1, 2, 1)
        for lr in lr_keys:
            losses = experiment_data["learning_rate"][lr][model]["losses"]["train"]
            ax1.plot(np.arange(1, len(losses) + 1), losses, label=lr)
        ax1.set_title("Left: Training Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(title="LR")
        # Validation loss
        ax2 = fig.add_subplot(1, 2, 2)
        for lr in lr_keys:
            losses = experiment_data["learning_rate"][lr][model]["losses"]["val"]
            ax2.plot(np.arange(1, len(losses) + 1), losses, label=lr)
        ax2.set_title("Right: Validation Loss Curves")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(title="LR")
        fig.suptitle(f"{model} Loss Curves on MNIST")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"MNIST_{model}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {model}: {e}")
        plt.close()

# Plot accuracy curves for MLP and CNN
for model in ["MLP", "CNN"]:
    try:
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        for lr in lr_keys:
            acc = experiment_data["learning_rate"][lr][model]["metrics"]["orig_acc"]
            ax1.plot(np.arange(1, len(acc) + 1), acc, label=lr)
        ax1.set_title("Left: Original Test Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(title="LR")
        ax2 = fig.add_subplot(1, 2, 2)
        for lr in lr_keys:
            acc = experiment_data["learning_rate"][lr][model]["metrics"]["aug_acc"]
            ax2.plot(np.arange(1, len(acc) + 1), acc, label=lr)
        ax2.set_title("Right: Augmented Test Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend(title="LR")
        fig.suptitle(f"{model} Accuracy Curves on MNIST")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(working_dir, f"MNIST_{model}_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {model}: {e}")
        plt.close()

# Plot CGR curves
try:
    plt.figure()
    for lr in lr_keys:
        cgr = experiment_data["learning_rate"][lr]["CGR"]
        plt.plot(np.arange(1, len(cgr) + 1), cgr, label=lr)
    plt.title("CGR over Epochs on MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("CGR")
    plt.legend(title="LR")
    plt.savefig(os.path.join(working_dir, "MNIST_CGR_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CGR plot: {e}")
    plt.close()
