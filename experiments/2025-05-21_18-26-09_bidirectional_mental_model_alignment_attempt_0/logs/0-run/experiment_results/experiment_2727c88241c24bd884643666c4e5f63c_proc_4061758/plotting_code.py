import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

datasets = list(data.keys())

# Plot 1: Loss curves comparison
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ds in datasets:
        tr, vl = data[ds]["losses"]["train"], data[ds]["losses"]["val"]
        epochs = range(1, len(tr) + 1)
        axes[0].plot(epochs, tr, label=ds)
        axes[1].plot(epochs, vl, label=ds)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    fig.suptitle(
        "All Datasets - Loss Curves\nLeft: Training Loss, Right: Validation Loss"
    )
    plt.savefig(os.path.join(working_dir, "all_datasets_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss comparison plot: {e}")
    plt.close()

# Plot 2: MAI curves comparison
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ds in datasets:
        tr, vl = data[ds]["metrics"]["train"], data[ds]["metrics"]["val"]
        epochs = range(1, len(tr) + 1)
        axes[0].plot(epochs, tr, label=ds)
        axes[1].plot(epochs, vl, label=ds)
    axes[0].set_title("Training MAI")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAI (1-JSD)")
    axes[0].legend()
    axes[1].set_title("Validation MAI")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAI (1-JSD)")
    axes[1].legend()
    fig.suptitle("All Datasets - MAI Curves\nLeft: Training MAI, Right: Validation MAI")
    plt.savefig(os.path.join(working_dir, "all_datasets_mai_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MAI comparison plot: {e}")
    plt.close()

# Plot 3: Accuracy bar chart
try:
    accuracies = []
    for ds in datasets:
        preds = data[ds]["predictions"]
        gts = data[ds]["ground_truth"]
        accuracies.append(np.mean(preds == gts))
    fig = plt.figure(figsize=(6, 4))
    plt.bar(datasets, accuracies)
    plt.ylim(0, 1)
    plt.title("All Datasets - Validation Accuracy\nBar Chart per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.savefig(os.path.join(working_dir, "all_datasets_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Print final evaluation metrics
for ds in datasets:
    final_loss = data[ds]["losses"]["val"][-1]
    final_mai = data[ds]["metrics"]["val"][-1]
    preds = data[ds]["predictions"]
    gts = data[ds]["ground_truth"]
    acc = np.mean(preds == gts)
    print(
        f"{ds}: Final Val Loss={final_loss:.4f}, Final Val MAI={final_mai:.4f}, Accuracy={acc:.4f}"
    )
