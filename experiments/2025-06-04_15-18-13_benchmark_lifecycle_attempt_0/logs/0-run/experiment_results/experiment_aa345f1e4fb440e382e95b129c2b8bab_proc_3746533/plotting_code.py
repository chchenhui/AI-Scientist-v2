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

# Print final test accuracies
if "activation_function_ablation" in experiment_data:
    for name, data in experiment_data["activation_function_ablation"].items():
        final_orig = data["metrics"]["orig_acc"][-1]
        final_aug = data["metrics"]["aug_acc"][-1]
        print(
            f"{name} - Final Original Acc: {final_orig:.4f}, Final Augmented Acc: {final_aug:.4f}"
        )

# Plot loss curves
try:
    plt.figure()
    plt.subplot(1, 2, 1)
    for name, d in experiment_data.get("activation_function_ablation", {}).items():
        plt.plot(d["losses"]["train"], label=name)
    plt.title("Training Loss\nDataset: MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    for name, d in experiment_data.get("activation_function_ablation", {}).items():
        plt.plot(d["losses"]["val"], label=name)
    plt.title("Validation Loss\nDataset: MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.suptitle("MNIST Activation Function Ablation")
    plt.savefig(os.path.join(working_dir, "MNIST_activation_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot accuracy curves
try:
    plt.figure()
    plt.subplot(1, 2, 1)
    for name, d in experiment_data.get("activation_function_ablation", {}).items():
        plt.plot(d["metrics"]["orig_acc"], label=name)
    plt.title("Original Test Accuracy\nDataset: MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    for name, d in experiment_data.get("activation_function_ablation", {}).items():
        plt.plot(d["metrics"]["aug_acc"], label=name)
    plt.title("Augmented Test Accuracy\nDataset: MNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.suptitle("MNIST Activation Function Ablation")
    plt.savefig(os.path.join(working_dir, "MNIST_activation_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves plot: {e}")
    plt.close()

# Plot final accuracy bar chart
try:
    names = list(experiment_data.get("activation_function_ablation", {}).keys())
    orig = [
        experiment_data["activation_function_ablation"][n]["metrics"]["orig_acc"][-1]
        for n in names
    ]
    aug = [
        experiment_data["activation_function_ablation"][n]["metrics"]["aug_acc"][-1]
        for n in names
    ]
    x = np.arange(len(names))
    w = 0.35
    plt.figure()
    plt.bar(x - w / 2, orig, w, label="Original")
    plt.bar(x + w / 2, aug, w, label="Augmented")
    plt.xticks(x, names, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Final Test Accuracies by Activation\nDataset: MNIST")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "MNIST_activation_final_accuracy_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar plot: {e}")
    plt.close()
