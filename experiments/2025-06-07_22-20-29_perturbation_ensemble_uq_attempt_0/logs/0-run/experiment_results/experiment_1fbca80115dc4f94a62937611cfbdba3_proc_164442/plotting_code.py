import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Print final detection metrics
for ablation, ds_dict in experiment_data.items():
    for dataset, data in ds_dict.items():
        try:
            last = data["metrics"]["detection"][-1]
            print(
                f"[{dataset}][{ablation}] Final metrics - "
                f"AUC_vote: {last['auc_vote']:.4f}, DES_vote: {last['DES_vote']:.4f}, "
                f"AUC_kl: {last['auc_kl']:.4f}, DES_kl: {last['DES_kl']:.4f}"
            )
        except Exception as e:
            print(f"Error printing metrics for {dataset}-{ablation}: {e}")

# Plot per dataset
for dataset in experiment_data.get("full_depth", {}):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ablation in ["full_depth", "reduced_depth"]:
            losses = experiment_data[ablation][dataset]["losses"]
            train = [x["loss"] for x in losses["train"]]
            val = [x["loss"] for x in losses["val"]]
            epochs = [x["epoch"] for x in losses["train"]]
            axes[0].plot(epochs, train, marker="o", label=f"{ablation}-train")
            axes[0].plot(epochs, val, marker="x", label=f"{ablation}-val")

            det = experiment_data[ablation][dataset]["metrics"]["detection"]
            ev = [m["auc_vote"] for m in det]
            ek = [m["auc_kl"] for m in det]
            ep = [m["epoch"] for m in det]
            axes[1].plot(ep, ev, marker="o", label=f"{ablation}-AUC_vote")
            axes[1].plot(ep, ek, marker="x", label=f"{ablation}-AUC_kl")

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Detection AUC (Vote vs KL)")
        axes[1].legend()
        fig.suptitle(
            f"{dataset} Experiment Results\n"
            "Left: Loss Curves, Right: Detection AUC (Vote vs KL)"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{dataset}_loss_detection.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {dataset}: {e}")
