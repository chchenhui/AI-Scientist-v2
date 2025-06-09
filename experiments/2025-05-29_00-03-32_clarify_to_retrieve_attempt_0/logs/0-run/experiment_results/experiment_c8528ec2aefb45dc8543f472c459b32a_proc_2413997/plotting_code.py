import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
data_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for ds_name, ds_data in experiment_data.get(
    "confidence_threshold_ablation", {}
).items():
    metrics = ds_data.get("metrics", {})
    thr = metrics.get("thresholds", [])
    base = metrics.get("baseline_acc", [])
    clar = metrics.get("clar_acc", [])
    turns = metrics.get("avg_turns", [])
    ces = metrics.get("CES", [])
    # Accuracy vs Threshold
    try:
        plt.figure()
        plt.plot(thr, base, marker="o", label="Baseline Acc")
        plt.plot(thr, clar, marker="x", label="Post-Clar Acc")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Accuracy vs Threshold")
        plt.legend()
        fname = f"{ds_name.replace(' ','')}_accuracy_vs_threshold.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()
    # Average Turns vs Threshold
    try:
        plt.figure()
        plt.plot(thr, turns, marker="s", color="g")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Avg Clar Turns")
        plt.title(f"{ds_name} Avg Turns vs Threshold")
        fname = f"{ds_name.replace(' ','')}_avg_turns_vs_threshold.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating avg turns plot for {ds_name}: {e}")
        plt.close()
    # CES vs Threshold
    try:
        plt.figure()
        plt.plot(thr, ces, marker="^", color="r")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("CES")
        plt.title(f"{ds_name} CES vs Threshold")
        fname = f"{ds_name.replace(' ','')}_CES_vs_threshold.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CES plot for {ds_name}: {e}")
        plt.close()
    # Loss curves if available
    losses = ds_data.get("losses", {})
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if train_loss and val_loss:
        try:
            plt.figure()
            plt.plot(train_loss, label="Train Loss")
            plt.plot(val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{ds_name} Training and Validation Loss")
            plt.legend()
            fname = f"{ds_name.replace(' ','')}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}: {e}")
            plt.close()
