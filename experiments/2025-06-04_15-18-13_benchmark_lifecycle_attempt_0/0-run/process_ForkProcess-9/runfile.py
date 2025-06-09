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

# Print final validation accuracies
for ds_name, data in experiment_data.items():
    for model_name, metrics in data["metrics"].items():
        val_acc = metrics["val_acc"][-1]
        print(f"{ds_name} - {model_name}: final validation accuracy = {val_acc:.4f}")

# Plot metrics curves per dataset
for ds_name, data in experiment_data.items():
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        epochs = np.arange(
            1, len(next(iter(data["metrics"].values()))["train_loss"]) + 1
        )
        for model_name, metrics in data["metrics"].items():
            ax1.plot(epochs, metrics["train_loss"], label=f"{model_name} train")
            ax1.plot(
                epochs, metrics["val_loss"], linestyle="--", label=f"{model_name} val"
            )
            ax2.plot(epochs, metrics["val_acc"], label=model_name)
        fig.suptitle(
            f"{ds_name.capitalize()} Metrics (Left: Loss Curves, Right: Accuracy Curves) Across Models"
        )
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_title("Accuracy Curves")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax1.legend()
        ax2.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_accuracy_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot for {ds_name}: {e}")
        plt.close()

# Plot discrimination score comparison across datasets
try:
    plt.figure()
    for ds_name, data in experiment_data.items():
        epochs = np.arange(1, len(data["discrimination_score"]) + 1)
        plt.plot(epochs, data["discrimination_score"], label=ds_name)
    plt.title("Discrimination Score Across Datasets")
    plt.xlabel("Epoch")
    plt.ylabel("Discrimination Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "discrimination_score_across_datasets.png"))
    plt.close()
except Exception as e:
    print(f"Error creating discrimination score plot: {e}")
    plt.close()

# Plot final validation accuracy comparison across datasets
try:
    labels = list(experiment_data.keys())
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    models = list(next(iter(experiment_data.values()))["metrics"].keys())
    for i, model_name in enumerate(models):
        accs = [
            experiment_data[ds]["metrics"][model_name]["val_acc"][-1] for ds in labels
        ]
        ax.bar(x + i * width, accs, width, label=model_name)
    ax.set_title("Final Validation Accuracy Comparison Across Datasets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(
        os.path.join(working_dir, "final_val_accuracy_comparison_across_datasets.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy comparison plot: {e}")
    plt.close()
