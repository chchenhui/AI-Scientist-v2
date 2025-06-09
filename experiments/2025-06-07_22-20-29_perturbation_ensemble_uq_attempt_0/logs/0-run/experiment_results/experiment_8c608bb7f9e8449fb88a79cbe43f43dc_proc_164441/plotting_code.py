import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")

# Load experiment data
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Iterate through ablations and datasets to plot
for ablation, ds_dict in experiment_data.items():
    for ds_name, ds_data in ds_dict.items():
        # Plot training vs validation loss
        try:
            plt.figure()
            epochs = [d["epoch"] for d in ds_data["losses"]["train"]]
            train_loss = [d["loss"] for d in ds_data["losses"]["train"]]
            val_loss = [d["loss"] for d in ds_data["losses"]["val"]]
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Loss Curves ({ablation})")
            plt.legend()
            fname = f"{ds_name}_{ablation}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating {ds_name} {ablation} loss plot: {e}")
            plt.close()
        # Plot detection AUC metrics side by side
        try:
            metrics = ds_data["metrics"]["detection"]
            epochs = [m["epoch"] for m in metrics]
            auc_vote = [m["auc_vote"] for m in metrics]
            auc_kl = [m["auc_kl"] for m in metrics]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(epochs, auc_vote, marker="o")
            axes[0].set_title("Vote AUC")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("AUC")
            axes[1].plot(epochs, auc_kl, marker="o")
            axes[1].set_title("KL AUC")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("AUC")
            plt.suptitle(
                f"{ds_name} Detection AUC ({ablation})\nLeft: Vote AUC, Right: KL AUC"
            )
            fname = f"{ds_name}_{ablation}_detection_auc.png"
            fig.savefig(os.path.join(working_dir, fname))
            plt.close(fig)
        except Exception as e:
            print(f"Error creating {ds_name} {ablation} detection plot: {e}")
            plt.close()
