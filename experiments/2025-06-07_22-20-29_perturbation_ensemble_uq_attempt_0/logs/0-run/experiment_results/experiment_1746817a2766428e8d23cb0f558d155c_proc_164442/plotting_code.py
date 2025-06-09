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
else:
    exp = experiment_data.get("No_Pretraining_RandomInit", {})
    datasets = list(exp.keys())
    # Print final detection AUC metrics
    for ds in datasets:
        det = exp[ds]["metrics"]["detection"]
        last = det[-1]
        print(
            f"{ds} Final AUC Vote: {last['auc_vote']:.4f}, AUC KL: {last['auc_kl']:.4f}"
        )
    # Plot loss curves
    try:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
        for ax, ds in zip(axes, datasets):
            losses = exp[ds]["losses"]
            epochs = [d["epoch"] for d in losses["train"]]
            tr = [d["loss"] for d in losses["train"]]
            va = [d["loss"] for d in losses["val"]]
            ax.plot(epochs, tr, label="Train Loss")
            ax.plot(epochs, va, label="Val Loss")
            ax.set_title(f"{ds} Loss Curves")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(working_dir, "all_datasets_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()
    # Plot detection AUC curves
    try:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
        for ax, ds in zip(axes, datasets):
            det = exp[ds]["metrics"]["detection"]
            epochs = [d["epoch"] for d in det]
            av = [d["auc_vote"] for d in det]
            ak = [d["auc_kl"] for d in det]
            ax.plot(epochs, av, label="AUC Vote")
            ax.plot(epochs, ak, label="AUC KL")
            ax.set_title(f"{ds} Detection AUC")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("AUC")
            ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(working_dir, "all_datasets_detection_auc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating detection AUC plots: {e}")
        plt.close()
