import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

warm_configs = sorted(experiment_data.get("lr_warmup", {}).keys(), key=lambda x: int(x))

# plot losses
try:
    plt.figure()
    for cfg in warm_configs:
        ed = experiment_data["lr_warmup"][cfg]["synthetic"]
        epochs = range(len(ed["losses"]["train"]))
        plt.plot(epochs, ed["losses"]["train"], marker="o", label=f"Warmup {cfg} Train")
        plt.plot(epochs, ed["losses"]["val"], marker="x", label=f"Warmup {cfg} Val")
    plt.title("Synthetic Dataset Loss vs Epoch for Different LR Warmup Steps")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_loss_vs_epoch_lr_warmup.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# plot memory retention metrics
try:
    plt.figure()
    for cfg in warm_configs:
        ed = experiment_data["lr_warmup"][cfg]["synthetic"]
        epochs = range(len(ed["metrics"]["train"]))
        plt.plot(
            epochs, ed["metrics"]["train"], marker="o", label=f"Warmup {cfg} Train"
        )
        plt.plot(epochs, ed["metrics"]["val"], marker="x", label=f"Warmup {cfg} Val")
    plt.title(
        "Synthetic Dataset Memory Retention Ratio vs Epoch for Different LR Warmup Steps"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Retention Ratio")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_ratio_vs_epoch_lr_warmup.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# plot predictions vs ground truth
try:
    n = len(warm_configs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2 * n))
    for i, cfg in enumerate(warm_configs):
        ed = experiment_data["lr_warmup"][cfg]["synthetic"]
        gt = ed["ground_truth"]
        preds = ed["predictions"]
        axes[i, 0].plot(gt, "o-")
        axes[i, 0].set_title(f"Warmup {cfg}: Ground Truth")
        axes[i, 0].set_ylabel("Token ID")
        axes[i, 1].plot(preds, "o-")
        axes[i, 1].set_title(f"Warmup {cfg}: Generated Samples")
    fig.suptitle("Synthetic Dataset: Left: Ground Truth, Right: Generated Samples")
    for ax in axes.flatten():
        ax.set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        os.path.join(working_dir, "synthetic_predictions_vs_ground_truth_lr_warmup.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()

# print final evaluation metrics
for cfg in warm_configs:
    ed = experiment_data["lr_warmup"][cfg]["synthetic"]
    final_val_loss = ed["losses"]["val"][-1]
    final_val_ratio = ed["metrics"]["val"][-1]
    print(
        f"Warmup {cfg}: Final Val Loss = {final_val_loss:.4f}, Final Val Retention Ratio = {final_val_ratio:.4f}"
    )
