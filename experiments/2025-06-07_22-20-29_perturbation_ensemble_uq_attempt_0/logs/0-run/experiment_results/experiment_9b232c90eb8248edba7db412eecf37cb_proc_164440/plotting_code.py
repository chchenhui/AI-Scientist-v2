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

for ds_name, ds_data in experiment_data.get("head_only", {}).items():
    # Loss curves
    try:
        losses = ds_data["losses"]
        epochs = [e["epoch"] for e in losses["train"]]
        train_l = [e["loss"] for e in losses["train"]]
        val_l = [e["loss"] for e in losses["val"]]
        plt.figure()
        plt.plot(epochs, train_l, label="Train Loss")
        plt.plot(epochs, val_l, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Loss Curves\nHead Only")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_head_only_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # Detection AUC curves
    try:
        metrics = ds_data["metrics"]["detection"]
        epochs = [m["epoch"] for m in metrics]
        auc_vote = [m["auc_vote"] for m in metrics]
        auc_kl = [m["auc_kl"] for m in metrics]
        plt.figure()
        plt.plot(epochs, auc_vote, label="AUC Vote")
        plt.plot(epochs, auc_kl, label="AUC KL")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"{ds_name} Detection AUC Curves\nHead Only")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_head_only_detection_auc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating detection AUC plot for {ds_name}: {e}")
        plt.close()

    # Detection DES curves
    try:
        des_vote = [m["DES_vote"] for m in metrics]
        des_kl = [m["DES_kl"] for m in metrics]
        plt.figure()
        plt.plot(epochs, des_vote, label="DES Vote")
        plt.plot(epochs, des_kl, label="DES KL")
        plt.xlabel("Epoch")
        plt.ylabel("Detection Score (normalized)")
        plt.title(f"{ds_name} Detection DES Curves\nHead Only")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_head_only_detection_des.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating detection DES plot for {ds_name}: {e}")
        plt.close()
