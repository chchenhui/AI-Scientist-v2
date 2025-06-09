import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["positional_embedding_ablation"]
    datasets = list(data.keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    datasets = []
    data = {}

# Print final evaluation metrics
for name in datasets:
    val_acc = data[name]["metrics"]["val"][-1]
    det = data[name]["detection"][-1]
    print(
        f"{name} - Final Val Acc: {val_acc:.4f}, AUC_vote: {det['auc_vote']:.4f}, AUC_kl: {det['auc_kl']:.4f}"
    )

# Plot 1: Loss curves
try:
    epochs = range(1, len(data[datasets[0]]["losses"]["train"]) + 1)
    plt.figure()
    for name in datasets:
        losses = data[name]["losses"]
        plt.plot(epochs, losses["train"], label=f"{name} Train")
        plt.plot(epochs, losses["val"], "--", label=f"{name} Val")
    plt.title("Loss Curves for sst2, yelp_polarity, imdb")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "loss_curves_positional_embedding_ablation.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# Plot 2: Accuracy curves
try:
    plt.figure()
    for name in datasets:
        mets = data[name]["metrics"]
        plt.plot(epochs, mets["train"], label=f"{name} Train")
        plt.plot(epochs, mets["val"], "--", label=f"{name} Val")
    plt.title("Accuracy Curves for sst2, yelp_polarity, imdb")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "accuracy_curves_positional_embedding_ablation.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# Plot 3: Detection AUC curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in datasets:
        dets = data[name]["detection"]
        ep = [d["epoch"] for d in dets]
        axes[0].plot(ep, [d["auc_vote"] for d in dets], label=name)
        axes[1].plot(ep, [d["auc_kl"] for d in dets], label=name)
    axes[0].set_title("AUC_vote over Epochs")
    axes[1].set_title("AUC_kl over Epochs")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.legend()
    fig.suptitle(
        "Detection AUC Curves for sst2, yelp_polarity, imdb\n(Left: AUC_vote, Right: AUC_kl)"
    )
    plt.savefig(
        os.path.join(
            working_dir, "detection_auc_curves_positional_embedding_ablation.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating detection AUC plots: {e}")
    plt.close()
