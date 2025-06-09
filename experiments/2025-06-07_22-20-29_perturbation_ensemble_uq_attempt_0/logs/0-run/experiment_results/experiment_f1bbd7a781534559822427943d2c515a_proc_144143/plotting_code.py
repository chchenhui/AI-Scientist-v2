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
    for dataset, exp in experiment_data.items():
        # find best hyperparams by max validation AUC
        best_entry = max(exp["metrics"]["val"], key=lambda x: x["auc"])
        bs, lr = best_entry["bs"], best_entry["lr"]
        print(f"{dataset}: Best Val AUC = {best_entry['auc']:.4f} at bs={bs}, lr={lr}")
        # collect losses
        loss_train = [
            (d["epoch"], d["loss"])
            for d in exp["losses"]["train"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        loss_val = [
            (d["epoch"], d["loss"])
            for d in exp["losses"]["val"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        loss_train.sort()
        loss_val.sort()
        epochs = [e for e, _ in loss_train]
        tr_loss = [l for _, l in loss_train]
        vl_loss = [l for _, l in loss_val]
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, vl_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dataset} Loss Curve (Train vs Validation)\nbs={bs}, lr={lr}")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_loss_curve_bs{bs}_lr{lr}.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dataset}: {e}")
            plt.close()
        # collect AUCs
        auc_train = [
            (d["epoch"], d["auc"])
            for d in exp["metrics"]["train"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        auc_val = [
            (d["epoch"], d["auc"])
            for d in exp["metrics"]["val"]
            if d["bs"] == bs and d["lr"] == lr
        ]
        auc_train.sort()
        auc_val.sort()
        epochs = [e for e, _ in auc_train]
        tr_auc = [a for _, a in auc_train]
        vl_auc = [a for _, a in auc_val]
        try:
            plt.figure()
            plt.plot(epochs, tr_auc, label="Train AUC")
            plt.plot(epochs, vl_auc, label="Val AUC")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.title(f"{dataset} AUC Curve (Train vs Validation)\nbs={bs}, lr={lr}")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{dataset}_auc_curve_bs{bs}_lr{lr}.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating AUC plot for {dataset}: {e}")
            plt.close()
