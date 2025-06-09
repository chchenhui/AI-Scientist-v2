import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}
synthetic = data.get("lr_scheduler_gamma", {}).get("synthetic", {})
if not synthetic:
    print("No synthetic experiment data found.")
else:
    gammas = sorted(synthetic.keys(), key=lambda x: float(x))
    for g in gammas:
        vals = synthetic[g]["metrics"]["val"]
        if vals:
            print(f"Gamma {g}: Best Val AUC = {max(vals):.4f}")

    try:
        plt.figure()
        for g in gammas:
            ep = range(1, len(synthetic[g]["losses"]["train"]) + 1)
            plt.plot(ep, synthetic[g]["losses"]["train"], label=f"{g} train")
            plt.plot(ep, synthetic[g]["losses"]["val"], label=f"{g} val")
        plt.title("Loss Curves on Synthetic dataset\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_lr_scheduler_gamma_loss_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        for g in gammas:
            ep = range(1, len(synthetic[g]["metrics"]["train"]) + 1)
            plt.plot(ep, synthetic[g]["metrics"]["train"], label=f"{g} train")
            plt.plot(ep, synthetic[g]["metrics"]["val"], label=f"{g} val")
        plt.title("ROC AUC Curves on Synthetic dataset\nTrain vs Validation")
        plt.xlabel("Epoch")
        plt.ylabel("ROC AUC")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, "synthetic_lr_scheduler_gamma_auc_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating AUC plot: {e}")
        plt.close()
