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

try:
    sd = experiment_data["num_heads_tuning"]["synthetic"]
    heads = sd["num_heads"]
    train_losses = sd["losses"]["train"]
    val_losses = sd["losses"]["val"]
    train_metrics = sd["metrics"]["train"]
    val_metrics = sd["metrics"]["val"]
    # Loss curves
    plt.figure()
    for h, tl, vl in zip(heads, train_losses, val_losses):
        plt.plot(tl, label=f"{h} heads train")
        plt.plot(vl, "--", label=f"{h} heads val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("synthetic Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    # Metric curves
    plt.figure()
    for h, tm, vm in zip(heads, train_metrics, val_metrics):
        plt.plot(tm, label=f"{h} heads train")
        plt.plot(vm, "--", label=f"{h} heads val")
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.title("synthetic Memory Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_retention_ratio.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# Sample predictions vs ground truth
preds = sd.get("predictions", [])
gts = sd.get("ground_truth", [])
for h, p, g in zip(heads, preds, gts):
    try:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(g, marker="o")
        plt.title("Ground Truth")
        plt.subplot(1, 2, 2)
        plt.plot(p, marker="o")
        plt.title("Generated Samples")
        plt.suptitle(f"synthetic Samples (num_heads={h})")
        fname = f"synthetic_samples_{h}heads.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating sample plot for {h} heads: {e}")
        plt.close()
