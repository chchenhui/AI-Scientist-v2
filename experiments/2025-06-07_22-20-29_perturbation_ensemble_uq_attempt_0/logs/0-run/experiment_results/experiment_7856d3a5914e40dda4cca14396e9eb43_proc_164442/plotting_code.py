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

for name, data in experiment_data.get("token_type_embedding_ablation", {}).items():
    # Extract curves and metrics
    train_entries = data["losses"]["train"]
    val_entries = data["losses"]["val"]
    epochs = [e["epoch"] for e in train_entries]
    train_losses = [e["loss"] for e in train_entries]
    val_losses = [e["loss"] for e in val_entries]
    metric_entries = data["metrics"]["val"]
    auc_vote = [m["auc_vote"] for m in metric_entries]
    auc_kl = [m["auc_kl"] for m in metric_entries]
    # Compute and print accuracy
    preds = np.array(data.get("predictions", []))
    gt = np.array(data.get("ground_truth", []))
    if preds.size and gt.size:
        acc = np.mean(preds == gt)
        print(f"{name} Validation Accuracy: {acc:.4f}")
    # Plot loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{name} - Loss Curves\nLeft: Training Loss, Right: Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {name}: {e}")
        plt.close()
    # Plot AUC metrics
    try:
        plt.figure()
        plt.plot(epochs, auc_vote, marker="o", label="AUC_vote")
        plt.plot(epochs, auc_kl, marker="s", label="AUC_kl")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"{name} - AUC Curves\nAUC_vote vs AUC_kl")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{name}_auc_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating AUC plot for {name}: {e}")
        plt.close()
