import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for ablation, settings in experiment_data.items():
    for key, res in settings.items():
        preds = res["predictions"]
        gt = res["ground_truth"]
        acc = np.mean(preds == gt)
        print(f"Ablation: {ablation}, Setting: {key}, Test Accuracy: {acc:.4f}")

for ablation in ["uniform", "thresholded", "confidence_weighted"]:
    key = "ai_bs_32_user_bs_32"
    try:
        train_acc = experiment_data[ablation][key]["metrics"]["train"]
        val_acc = experiment_data[ablation][key]["metrics"]["val"]
        plt.figure()
        plt.plot(
            np.arange(1, len(train_acc) + 1), train_acc, label="Train", linestyle="-"
        )
        plt.plot(
            np.arange(1, len(val_acc) + 1), val_acc, label="Validation", linestyle="--"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.suptitle("Dataset: Synthetic Binary Classification")
        plt.title(
            f"{ablation.capitalize()} Ablation - Solid: Train, Dashed: Validation"
        )
        fname = f"synthetic_{ablation}_accuracy_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {ablation} accuracy plot: {e}")
        plt.close()
