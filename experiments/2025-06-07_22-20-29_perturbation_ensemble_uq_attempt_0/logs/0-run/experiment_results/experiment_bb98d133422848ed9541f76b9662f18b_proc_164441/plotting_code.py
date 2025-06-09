import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

for ablation, datasets in exp.items():
    for ds_name, data in datasets.items():
        # Loss curves
        try:
            tr = data["losses"]["train"]
            vl = data["losses"]["val"]
            epochs = [e["epoch"] for e in tr]
            plt.figure()
            plt.plot(epochs, [e["loss"] for e in tr], label="Train Loss")
            plt.plot(epochs, [e["loss"] for e in vl], label="Val Loss")
            plt.title(f"Loss Curves - {ds_name} ({ablation})\nTraining vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_{ablation}_loss_curve.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name} {ablation}: {e}")
            plt.close()
        # Detection AUC curves
        try:
            det = data["metrics"]["detection"]
            epochs = [m["epoch"] for m in det]
            plt.figure()
            plt.plot(epochs, [m["auc_vote"] for m in det], label="AUC Vote")
            plt.plot(epochs, [m["auc_kl"] for m in det], label="AUC KL")
            plt.title(f"Detection AUC Curves - {ds_name} ({ablation})\nVote vs KL")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{ds_name}_{ablation}_auc_curve.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error creating detection plot for {ds_name} {ablation}: {e}")
            plt.close()
