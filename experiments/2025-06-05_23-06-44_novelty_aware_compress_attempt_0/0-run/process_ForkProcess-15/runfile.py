import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

# Plotting loops
for ablation, ab_data in exp.items():
    for ds, ds_data in ab_data.items():
        # Loss curves
        try:
            tr = ds_data["losses"]["train"]
            va = ds_data["losses"]["val"]
            epochs = np.arange(1, len(tr) + 1)
            plt.figure()
            plt.suptitle(f"{ds} - {ablation.title()} Ablation")
            plt.title("Loss over Epochs")
            plt.plot(epochs, tr, marker="o", label="Train")
            plt.plot(epochs, va, marker="o", linestyle="--", label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            fname = f"{ablation}_{ds}_loss.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ablation}, {ds}: {e}")
            plt.close()

        # Metric curves
        for metric, splits in ds_data.get("metrics", {}).items():
            try:
                trm = splits["train"]
                vam = splits["val"]
                epochs = np.arange(1, len(trm) + 1)
                plt.figure()
                plt.suptitle(f"{ds} - {ablation.title()} Ablation")
                plt.title(metric)
                plt.plot(epochs, trm, marker="o", label="Train")
                plt.plot(epochs, vam, marker="o", linestyle="--", label="Val")
                plt.xlabel("Epoch")
                plt.ylabel(metric)
                plt.legend()
                metric_clean = metric.lower().replace(" ", "_")
                fname = f"{ablation}_{ds}_{metric_clean}.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
            except Exception as e:
                print(f"Error creating {metric} plot for {ablation}, {ds}: {e}")
                plt.close()
