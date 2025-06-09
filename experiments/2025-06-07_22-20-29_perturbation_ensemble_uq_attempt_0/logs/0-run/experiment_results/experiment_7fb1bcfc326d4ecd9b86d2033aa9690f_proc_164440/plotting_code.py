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

for ablation_key, ablation_data in experiment_data.items():
    for ds_name, ds_data in ablation_data.items():
        # Loss curves
        try:
            plt.figure()
            epochs = list(range(1, len(ds_data["losses"]["train"]) + 1))
            plt.plot(epochs, ds_data["losses"]["train"], label="Train Loss")
            plt.plot(epochs, ds_data["losses"]["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"Loss Curves — {ds_name}\nTraining vs Validation Loss on {ds_name}"
            )
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}: {e}")
            plt.close()
        # AUC curves
        try:
            plt.figure()
            metrics = ds_data["metrics"]["val"]
            epochs = [m["epoch"] for m in metrics]
            auc_vote = [m["auc_vote"] for m in metrics]
            auc_kl = [m["auc_kl"] for m in metrics]
            plt.plot(epochs, auc_vote, marker="o", label="AUC_vote")
            plt.plot(epochs, auc_kl, marker="s", label="AUC_kl")
            plt.xlabel("Epoch")
            plt.ylabel("AUC")
            plt.title(f"AUC Curves — {ds_name}\nAUC_vote and AUC_kl on {ds_name}")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_auc_curve.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating AUC plot for {ds_name}: {e}")
            plt.close()
        # Label distribution
        try:
            plt.figure()
            gt = ds_data["ground_truth"]
            pred = ds_data["predictions"]
            counts_gt = [gt.count(0), gt.count(1)]
            counts_pred = [pred.count(0), pred.count(1)]
            x = np.arange(2)
            width = 0.35
            plt.bar(x - width / 2, counts_gt, width, label="Ground Truth")
            plt.bar(x + width / 2, counts_pred, width, label="Predictions")
            plt.xticks(x, ["Class 0", "Class 1"])
            plt.ylabel("Count")
            plt.title(
                f"Label Distribution — {ds_name}\nLeft: Ground Truth, Right: Predictions for {ds_name}"
            )
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_label_distribution.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating label distribution plot for {ds_name}: {e}")
            plt.close()

# Print final metrics
for ablation_key, ablation_data in experiment_data.items():
    for ds_name, ds_data in ablation_data.items():
        final = ds_data["metrics"]["val"][-1]
        print(
            f"{ds_name} final metrics: AUC_vote={final['auc_vote']:.4f}, DES_vote={final['DES_vote']:.4f}, "
            f"AUC_kl={final['auc_kl']:.4f}, DES_kl={final['DES_kl']:.4f}"
        )
