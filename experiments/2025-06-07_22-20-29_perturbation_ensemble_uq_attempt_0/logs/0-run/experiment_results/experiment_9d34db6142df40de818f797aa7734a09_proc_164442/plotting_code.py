import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
# Load experiment data
try:
    data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = data.get("No_Dropout_Ablation", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

for ds_name, expd in exp.items():
    # Prepare loss data
    train_vals = expd["losses"]["train"]
    val_vals = expd["losses"]["val"]
    epochs = [e["epoch"] for e in train_vals]
    train_loss = [e["loss"] for e in train_vals]
    val_loss = [e["loss"] for e in val_vals]
    # Plot loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, marker="o", label="Train Loss")
        plt.plot(epochs, val_loss, marker="o", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Loss Curves (Training & Validation)")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fn)
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
    finally:
        plt.close()
    # Prepare detection metrics
    det = expd["metrics"]["detection"]
    det_epochs = [m["epoch"] for m in det]
    auc_vote = [m["auc_vote"] for m in det]
    auc_kl = [m["auc_kl"] for m in det]
    # Plot detection AUCs
    try:
        plt.figure()
        plt.plot(det_epochs, auc_vote, marker="o", label="AUC_vote")
        plt.plot(det_epochs, auc_kl, marker="o", label="AUC_kl")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"{ds_name} Detection AUC Metrics")
        plt.legend()
        fn2 = os.path.join(working_dir, f"{ds_name}_detection_auc.png")
        plt.savefig(fn2)
    except Exception as e:
        print(f"Error creating detection plot for {ds_name}: {e}")
    finally:
        plt.close()
    # Print final metrics
    if det:
        last = det[-1]
        print(
            f"{ds_name} Final AUC_vote: {last['auc_vote']:.4f}, AUC_kl: {last['auc_kl']:.4f}"
        )
