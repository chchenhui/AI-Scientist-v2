import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    metrics = data.get("always_ask_clar", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    metrics = {}

datasets = list(metrics.keys())
baseline = [metrics[d]["metrics"]["baseline_acc"] for d in datasets]
clar = [metrics[d]["metrics"]["clar_acc"] for d in datasets]
ces = [metrics[d]["metrics"]["CES"] for d in datasets]

# Accuracy comparison bar chart
try:
    plt.figure()
    x = np.arange(len(datasets))
    width = 0.35
    plt.bar(x - width / 2, baseline, width, label="Baseline")
    plt.bar(x + width / 2, clar, width, label="Post-Clarification")
    plt.xticks(x, datasets, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison\nLeft: Baseline, Right: Post-Clarification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "accuracy_comparison_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy comparison plot: {e}")
    plt.close()

# CES comparison bar chart
try:
    plt.figure()
    plt.bar(datasets, ces)
    plt.ylabel("CES")
    plt.title("Comparison Efficiency Score (CES)\nAcross Datasets")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "ces_comparison_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CES plot: {e}")
    plt.close()

# Training/validation loss curves if available
try:
    for d in datasets:
        loss_dict = metrics[d].get("losses", {})
        train = loss_dict.get("train", [])
        val = loss_dict.get("val", [])
        if train or val:
            plt.figure()
            epochs = np.arange(1, max(len(train), len(val)) + 1)
            if train:
                plt.plot(np.arange(1, len(train) + 1), train, label="Train Loss")
            if val:
                plt.plot(np.arange(1, len(val) + 1), val, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{d} Loss Curves\nTraining and Validation")
            plt.legend()
            plt.tight_layout()
            fname = f"{d.lower().replace(' ', '_')}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
except Exception as e:
    print(f"Error creating loss plots: {e}")
    plt.close()
