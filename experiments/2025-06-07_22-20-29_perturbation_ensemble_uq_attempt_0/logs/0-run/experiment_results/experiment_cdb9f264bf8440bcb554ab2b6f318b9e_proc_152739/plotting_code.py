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
else:
    # Per‐dataset visualizations
    for dataset, exp in experiment_data.items():
        # Loss curves
        try:
            epochs = [d["epoch"] for d in exp["losses"]["train"]]
            tr_loss = [d["loss"] for d in exp["losses"]["train"]]
            vl_loss = [d["loss"] for d in exp["losses"]["val"]]
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, vl_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} Loss Curve (Train vs Validation)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dataset}: {e}")
            plt.close()
        # Detection AUC curve
        try:
            epochs = [d["epoch"] for d in exp["metrics"]["detection"]]
            auc = [d["auc"] for d in exp["metrics"]["detection"]]
            plt.figure()
            plt.plot(epochs, auc, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Detection AUC")
            plt.title(f"{dataset} Detection AUC Curve")
            plt.savefig(os.path.join(working_dir, f"{dataset}_detection_auc_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating detection AUC plot for {dataset}: {e}")
            plt.close()
        # Class distribution vs ground truth
        try:
            preds = exp["predictions"]
            gt = list(exp["ground_truth"])
            counts_pred = [preds.count(0), preds.count(1)]
            counts_gt = [gt.count(0), gt.count(1)]
            x = np.arange(2)
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, counts_gt, width, label="Ground Truth")
            plt.bar(x + width / 2, counts_pred, width, label="Predictions")
            plt.xticks(x, ["Class 0", "Class 1"])
            plt.ylabel("Count")
            plt.title(f"{dataset} Class Distribution: GT vs Preds")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_class_distribution.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating class distribution plot for {dataset}: {e}")
            plt.close()

    # Combined detection AUC comparison
    try:
        plt.figure()
        for dataset, exp in experiment_data.items():
            epochs = [d["epoch"] for d in exp["metrics"]["detection"]]
            auc = [d["auc"] for d in exp["metrics"]["detection"]]
            plt.plot(epochs, auc, marker="o", label=dataset)
        plt.xlabel("Epoch")
        plt.ylabel("Detection AUC")
        plt.title("Comparison of Detection AUC Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_detection_auc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating combined detection AUC plot: {e}")
        plt.close()

    # Combined DES comparison
    try:
        plt.figure()
        for dataset, exp in experiment_data.items():
            epochs = [d["epoch"] for d in exp["metrics"]["detection"]]
            des = [d["DES"] for d in exp["metrics"]["detection"]]
            plt.plot(epochs, des, marker="o", label=dataset)
        plt.xlabel("Epoch")
        plt.ylabel("DES")
        plt.title("Comparison of DES Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_DES.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating combined DES plot: {e}")
        plt.close()
