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
    # Print final detection AUC for each dataset
    for dataset, exp in experiment_data.items():
        try:
            final_auc = exp["metrics"]["detection"][-1]["auc"]
            print(f"{dataset}: Final Detection AUC = {final_auc:.4f}")
        except Exception as e:
            print(f"Error retrieving final AUC for {dataset}: {e}")
    # Per-dataset plots
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
            plt.title(f"Train vs Validation Loss\nDataset: {dataset}")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dataset}_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dataset}: {e}")
            plt.close()
        # Detection AUC curves
        try:
            epochs = [d["epoch"] for d in exp["metrics"]["detection"]]
            aucs = [d["auc"] for d in exp["metrics"]["detection"]]
            plt.figure()
            plt.plot(epochs, aucs, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Detection AUC")
            plt.title(f"Detection AUC over Epochs\nDataset: {dataset}")
            plt.savefig(os.path.join(working_dir, f"{dataset}_detection_auc_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating detection AUC plot for {dataset}: {e}")
            plt.close()
    # Comparison plot across datasets
    try:
        plt.figure()
        for dataset, exp in experiment_data.items():
            epochs = [d["epoch"] for d in exp["metrics"]["detection"]]
            aucs = [d["auc"] for d in exp["metrics"]["detection"]]
            plt.plot(epochs, aucs, marker="o", label=dataset)
        plt.xlabel("Epoch")
        plt.ylabel("Detection AUC")
        plt.title("Detection AUC Comparison\nAcross Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "detection_auc_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()
