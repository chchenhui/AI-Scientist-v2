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
    experiment_data = {}

for ds in experiment_data.get("full_heads", {}):
    try:
        data_plot = {}
        for exp in ["full_heads", "pruned_heads"]:
            ed = experiment_data[exp][ds]
            epochs = [e["epoch"] for e in ed["metrics"]["train"]]
            loss_tr = [e["loss"] for e in ed["losses"]["train"]]
            loss_va = [e["loss"] for e in ed["losses"]["val"]]
            acc_tr = [e["acc"] for e in ed["metrics"]["train"]]
            acc_va = [e["acc"] for e in ed["metrics"]["val"]]
            auc_vote = [e["auc_vote"] for e in ed["detection"]]
            auc_kl = [e["auc_kl"] for e in ed["detection"]]
            data_plot[exp] = dict(
                epochs=epochs,
                loss_tr=loss_tr,
                loss_va=loss_va,
                acc_tr=acc_tr,
                acc_va=acc_va,
                auc_vote=auc_vote,
                auc_kl=auc_kl,
            )
        plt.figure(figsize=(8, 12))
        plt.suptitle(f"Dataset: {ds}")
        # Loss curves
        plt.subplot(3, 1, 1)
        for exp, dp in data_plot.items():
            lbl = exp.replace("_", " ").title()
            plt.plot(dp["epochs"], dp["loss_tr"], marker="o", label=f"{lbl} Train")
            plt.plot(
                dp["epochs"],
                dp["loss_va"],
                marker="o",
                linestyle="--",
                label=f"{lbl} Val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Val Loss")
        plt.legend()
        # Accuracy curves
        plt.subplot(3, 1, 2)
        for exp, dp in data_plot.items():
            lbl = exp.replace("_", " ").title()
            plt.plot(dp["epochs"], dp["acc_tr"], marker="o", label=f"{lbl} Train")
            plt.plot(
                dp["epochs"],
                dp["acc_va"],
                marker="o",
                linestyle="--",
                label=f"{lbl} Val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Train vs Val Accuracy")
        plt.legend()
        # Detection AUC curves
        plt.subplot(3, 1, 3)
        for exp, dp in data_plot.items():
            lbl = exp.replace("_", " ").title()
            plt.plot(dp["epochs"], dp["auc_vote"], marker="o", label=f"{lbl} Vote AUC")
            plt.plot(
                dp["epochs"],
                dp["auc_kl"],
                marker="o",
                linestyle="--",
                label=f"{lbl} KL AUC",
            )
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title("Detection AUC Metrics")
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(working_dir, f"{ds}_summary.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {ds}: {e}")
        plt.close()
