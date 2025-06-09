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

for ds_name, ds_exp in experiment_data.items():
    try:
        epochs = [d["epoch"] for d in ds_exp["losses"]["train"]]
        train_loss = [d["loss"] for d in ds_exp["losses"]["train"]]
        val_loss = [d["loss"] for d in ds_exp["losses"]["val"]]
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curve\n{ds_name} Dataset: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    try:
        epochs = [d["epoch"] for d in ds_exp["metrics"]["val"]]
        auc_v = [d["auc_vote"] for d in ds_exp["metrics"]["val"]]
        auc_k = [d["auc_kl"] for d in ds_exp["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, auc_v, label="AUC Vote")
        plt.plot(epochs, auc_k, label="AUC KL")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"AUC Metrics\n{ds_name} Dataset: Vote vs KL")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_auc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating AUC plot for {ds_name}: {e}")
        plt.close()
