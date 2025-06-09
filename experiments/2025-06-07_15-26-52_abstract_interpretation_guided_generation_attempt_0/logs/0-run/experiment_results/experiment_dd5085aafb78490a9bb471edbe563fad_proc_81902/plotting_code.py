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

# Dataset-specific plots
for ds, data in experiment_data.items():
    losses = data.get("losses", {})
    train_loss = np.array(losses.get("train", []))
    val_loss = np.array(losses.get("val", []))
    epochs = np.arange(1, len(train_loss) + 1)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, train_loss)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].plot(epochs, val_loss)
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        fig.suptitle(f"{ds} Loss Curves\nLeft: Training Loss, Right: Validation Loss")
        fig.savefig(os.path.join(working_dir, f"{ds}_loss_curves.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating {ds} loss curves plot: {e}")
        plt.close()

    metrics = data.get("metrics", {})
    val_acc = np.array(metrics.get("val_acc", []))
    cer = np.array(metrics.get("constraint_effectiveness_rate", []))
    epochs_m = np.arange(1, len(val_acc) + 1)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs_m, val_acc)
        axes[0].set_title("Validation Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[1].plot(epochs_m, cer)
        axes[1].set_title("Constraint Effectiveness Rate")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("CER")
        fig.suptitle(
            f"{ds} Validation Metrics\nLeft: Accuracy, Right: Constraint Effectiveness Rate"
        )
        fig.savefig(os.path.join(working_dir, f"{ds}_val_metrics.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error creating {ds} validation metrics plot: {e}")
        plt.close()

# Cross-dataset comparison plot
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ds, data in experiment_data.items():
        vm = data.get("metrics", {})
        acc = np.array(vm.get("val_acc", []))
        cr = np.array(vm.get("constraint_effectiveness_rate", []))
        epochs_acc = np.arange(1, len(acc) + 1)
        epochs_cr = np.arange(1, len(cr) + 1)
        axes[0].plot(epochs_acc, acc, label=ds)
        axes[1].plot(epochs_cr, cr, label=ds)
    axes[0].set_title("Validation Accuracy Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(title="Dataset")
    axes[1].set_title("Constraint Effectiveness Rate Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CER")
    axes[1].legend(title="Dataset")
    fig.suptitle(
        "Validation Metrics Comparison\nLeft: Accuracy, Right: Constraint Effectiveness Rate Across Datasets"
    )
    fig.savefig(os.path.join(working_dir, "comparison_val_metrics.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating cross-dataset comparison plot: {e}")
    plt.close()
